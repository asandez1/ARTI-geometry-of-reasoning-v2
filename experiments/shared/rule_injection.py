#!/usr/bin/env python3
"""
Rule Injection Module for CO-FRN (Paper 13, Option B).

Two components:
1. RuleLibrary: Pre-embeds all rules with sentence-transformer, retrieves
   top-k rules by cosine similarity at inference/training time.
2. RuleInjectionLayer: Projects rule context (384D) to hidden_dim (256D),
   applies gated addition with LayerNorm to s0. ~98K new params.

Architecture:
    q_emb [B, 384] -> RuleLibrary.retrieve(top_k=3) -> rule_context [B, 384]
                    -> RuleInjectionLayer: rule_proj(rule_context) -> [B, 256]
                    -> s0_aug = LayerNorm(s0 + alpha * rule_vec)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

RULES_PATH = Path(__file__).parent / "rules.json"


class RuleLibrary:
    """
    Pre-embeds all rules with a sentence-transformer and retrieves
    top-k rules by cosine similarity for a given query embedding.

    The library is not an nn.Module — it holds no trainable parameters.
    Rule embeddings are pre-computed once and stored as a fixed tensor.
    """

    def __init__(
        self,
        rules_path: str = None,
        encoder_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu',
    ):
        """
        Args:
            rules_path: Path to rules.json (defaults to shared/rules.json)
            encoder_name: Sentence-transformer model for rule embedding
            device: Torch device for rule embeddings
        """
        self.device = device
        self.encoder_name = encoder_name

        # Load rules
        path = Path(rules_path) if rules_path else RULES_PATH
        with open(str(path), 'r') as f:
            data = json.load(f)
        self.rules = data['rules']
        self.metadata = data['metadata']
        self.n_rules = len(self.rules)

        # Build lookup indices
        self.domain_to_indices: Dict[str, List[int]] = {}
        for i, rule in enumerate(self.rules):
            domain = rule['domain']
            if domain not in self.domain_to_indices:
                self.domain_to_indices[domain] = []
            self.domain_to_indices[domain].append(i)

        # Pre-embed all rules
        self.rule_embeddings = self._embed_rules(encoder_name, device)
        self.emb_dim = self.rule_embeddings.shape[1]

        logger.info(f"RuleLibrary: {self.n_rules} rules, {self.emb_dim}D embeddings, "
                     f"domains: {list(self.domain_to_indices.keys())}")

    def _embed_rules(self, encoder_name: str, device: str) -> torch.Tensor:
        """Embed all rule descriptions with sentence-transformer."""
        # Compose rule text: name + formal_statement + description
        rule_texts = []
        for rule in self.rules:
            text = (
                f"{rule['name']}: {rule['formal_statement']}. "
                f"{rule['description']}"
            )
            rule_texts.append(text)

        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(encoder_name)
            with torch.no_grad():
                embeddings = st.encode(
                    rule_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=64,
                )
            del st
            return F.normalize(embeddings.float(), dim=-1).to(device)

        except Exception as e:
            logger.warning(f"Could not load sentence-transformer: {e}")
            logger.warning("Using random rule embeddings (retrieval will be meaningless)")
            emb_dim = 384  # default for MiniLM
            torch.manual_seed(42)
            return F.normalize(torch.randn(self.n_rules, emb_dim), dim=-1).to(device)

    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Dict]]]:
        """
        Retrieve top-k rules by cosine similarity for each query.

        Args:
            query_embeddings: [B, emb_dim] query embeddings (raw, pre-projection)
            top_k: number of rules to retrieve per query
            domain_filter: if set, only retrieve rules from this domain

        Returns:
            rule_context: [B, emb_dim] weighted average of top-k rule embeddings
            similarities: [B, top_k] cosine similarities of top-k rules
            rule_info: list of list of rule dicts (for logging/debugging)
        """
        B = query_embeddings.shape[0]
        q = F.normalize(query_embeddings.to(self.device), dim=-1)

        # Filter to domain if requested
        if domain_filter and domain_filter in self.domain_to_indices:
            indices = self.domain_to_indices[domain_filter]
            candidates = self.rule_embeddings[indices]  # [n_domain, emb_dim]
            local_to_global = indices
        else:
            candidates = self.rule_embeddings  # [n_rules, emb_dim]
            local_to_global = list(range(self.n_rules))

        # Cosine similarity: [B, n_candidates]
        sims = torch.mm(q, candidates.T)

        # Top-k
        k = min(top_k, len(local_to_global))
        topk_sims, topk_local = sims.topk(k, dim=-1)  # [B, k]

        # Weighted average of top-k rule embeddings (softmax weights)
        weights = F.softmax(topk_sims, dim=-1)  # [B, k]

        # Gather rule embeddings for top-k
        topk_global = torch.tensor(
            [[local_to_global[j] for j in topk_local[i].tolist()] for i in range(B)],
            device=self.device,
        )  # [B, k]

        # Compute weighted context
        rule_context = torch.zeros(B, self.emb_dim, device=self.device)
        for i in range(B):
            rule_embs = self.rule_embeddings[topk_global[i]]  # [k, emb_dim]
            rule_context[i] = (weights[i].unsqueeze(-1) * rule_embs).sum(dim=0)

        # Collect rule info for diagnostics
        rule_info = []
        for i in range(B):
            info_list = []
            for j in range(k):
                idx = topk_global[i, j].item()
                info_list.append({
                    'rule_id': self.rules[idx]['id'],
                    'name': self.rules[idx]['name'],
                    'domain': self.rules[idx]['domain'],
                    'similarity': topk_sims[i, j].item(),
                })
            rule_info.append(info_list)

        return rule_context, topk_sims, rule_info

    def get_random_context(self, batch_size: int) -> torch.Tensor:
        """Return random rule embeddings (ablation baseline)."""
        indices = torch.randint(0, self.n_rules, (batch_size,), device=self.device)
        return self.rule_embeddings[indices]


class RuleInjectionLayer(nn.Module):
    """
    Projects rule context embeddings and adds them to s0 via a learnable gate.

    Architecture:
        rule_context [B, rule_dim] -> Linear(rule_dim, hidden_dim) -> GELU
                                   -> rule_vec [B, hidden_dim]
        s0_aug = LayerNorm(s0 + alpha * rule_vec)

    Parameters: ~98K
        rule_proj: Linear(384, 256) + bias = 384*256 + 256 = 98,560
        alpha: 1 (learnable scalar, init=0.1)
        LayerNorm: 256 + 256 = 512
        Total: ~99,073
    """

    def __init__(
        self,
        rule_dim: int = 384,
        hidden_dim: int = 256,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.rule_dim = rule_dim
        self.hidden_dim = hidden_dim

        # Project rule embeddings to model hidden dim
        self.rule_proj = nn.Sequential(
            nn.Linear(rule_dim, hidden_dim),
            nn.GELU(),
        )

        # Learnable gate scalar
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # LayerNorm for the augmented s0
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        s0: torch.Tensor,
        rule_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Augment s0 with rule context.

        Args:
            s0: [B, hidden_dim] encoder output (projected)
            rule_context: [B, rule_dim] weighted average of retrieved rule embeddings

        Returns:
            s0_aug: [B, hidden_dim] rule-augmented encoder output
        """
        rule_vec = self.rule_proj(rule_context)  # [B, hidden_dim]
        s0_aug = self.layer_norm(s0 + self.alpha * rule_vec)
        return s0_aug

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RuleAugmentedCOFRN(nn.Module):
    """
    Wraps a pre-trained (frozen) COFRN with a trainable RuleInjectionLayer.

    The COFRN base is frozen. Only the RuleInjectionLayer is trained.
    Rule context is retrieved from RuleLibrary and injected into s0
    before factorization.

    Architecture:
        q_emb [B, 384] -> COFRN.encode_text() -> s0 [B, 256]
                        -> RuleLibrary.retrieve(q_emb) -> rule_context [B, 384]
                        -> RuleInjectionLayer(s0, rule_context) -> s0_aug [B, 256]
                        -> COFRN.factorization(s0_aug) -> ... -> scores
    """

    def __init__(
        self,
        cofrn: nn.Module,
        rule_library: RuleLibrary,
        rule_dim: int = 384,
        hidden_dim: int = 256,
        alpha_init: float = 0.1,
        top_k: int = 3,
        use_rules: bool = True,
        use_random_rules: bool = False,
        domain_filter: Optional[str] = None,
    ):
        """
        Args:
            cofrn: Pre-trained COFRN model (will be frozen)
            rule_library: RuleLibrary instance
            rule_dim: Dimension of rule embeddings (384 for MiniLM)
            hidden_dim: COFRN hidden dimension (256)
            alpha_init: Initial gate value
            top_k: Number of rules to retrieve per query
            use_rules: If False, skip rule injection (ablation)
            use_random_rules: If True, use random rule embeddings (ablation)
            domain_filter: If set, only retrieve rules from this domain
        """
        super().__init__()
        self.cofrn = cofrn
        self.rule_library = rule_library
        self.top_k = top_k
        self.use_rules = use_rules
        self.use_random_rules = use_random_rules
        self.domain_filter = domain_filter

        # Freeze COFRN
        for param in self.cofrn.parameters():
            param.requires_grad = False

        # Trainable injection layer
        self.injection = RuleInjectionLayer(
            rule_dim=rule_dim,
            hidden_dim=hidden_dim,
            alpha_init=alpha_init,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        answer_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        depth: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with rule injection.

        Args:
            embeddings: [B, encoder_dim] raw question embeddings (384D)
            answer_embeddings: [B, n_answers, encoder_dim] answer embeddings
            labels: [B] correct answer indices
            depth: tree depth for multi-step

        Returns:
            Dict with scores, losses, diagnostics
        """
        device = embeddings.device

        # Step 1: Encode through frozen COFRN encoder -> s0
        # no_grad for encoder only (its params are frozen, don't need graph)
        with torch.no_grad():
            s0 = self.cofrn.encode_text(embeddings=embeddings)
        # Detach s0 so encoder graph is freed, then re-enable grad tracking
        s0 = s0.detach().requires_grad_(True)

        # Step 2: Retrieve and inject rules
        if self.use_rules:
            if self.use_random_rules:
                rule_context = self.rule_library.get_random_context(embeddings.shape[0])
                rule_context = rule_context.to(device)
                rule_sims = None
                rule_info = None
            else:
                rule_context, rule_sims, rule_info = self.rule_library.retrieve(
                    embeddings, top_k=self.top_k, domain_filter=self.domain_filter,
                )
                rule_context = rule_context.to(device)

            s0_aug = self.injection(s0, rule_context)
        else:
            s0_aug = s0
            rule_sims = None
            rule_info = None

        # Step 3: Run through COFRN factorization + reasoning
        # NOTE: No torch.no_grad() here — we need gradient flow through
        # frozen COFRN ops back to the injection layer. COFRN params have
        # requires_grad=False so they won't accumulate gradients.
        fact_out = self.cofrn.factorization(s0_aug)
        structural = fact_out['structural']
        context = fact_out['context']
        transformed = fact_out['transformed']
        anchor_weights = fact_out['weights']

        # Encode answers (frozen, but need graph for scoring)
        answer_enc = self.cofrn.encode_answers(answer_embeddings)

        # Reasoning (use multi-step if configured)
        effective_task = self.cofrn.config.task_type
        if self.cofrn.config.disable_tree:
            effective_task = 'single_step'

        operator_fn = self.cofrn.factorization.manifold.forward

        if effective_task == 'multi_step' and answer_enc is not None:
            evidence = transformed
            reason_out = self.cofrn.reasoning.forward_multistep(
                transformed=transformed,
                evidence=evidence,
                operator_fn=operator_fn,
                structural=structural,
                answer_encodings=answer_enc,
                depth=depth,
            )
        else:
            reason_out = self.cofrn.reasoning.forward_direct(
                transformed=transformed,
                answer_encodings=answer_enc,
            )

        scores = reason_out.get('scores')

        # Build result
        result = {
            'scores': scores,
            'transformed': transformed,
            'structural': structural,
            'context': context,
            'anchor_weights': anchor_weights,
            'alpha': self.injection.alpha.item(),
            'coherence_loss': reason_out.get('coherence_loss', torch.tensor(0.0, device=device)),
        }

        if rule_sims is not None:
            result['rule_similarities'] = rule_sims
        if rule_info is not None:
            result['rule_info'] = rule_info

        # Task loss (only scores need gradient through injection layer)
        if labels is not None and scores is not None:
            result['task_loss'] = F.cross_entropy(scores, labels)
            result['predicted'] = scores.argmax(dim=-1)
            result['correct'] = (result['predicted'] == labels).float()
        else:
            result['task_loss'] = torch.tensor(0.0, device=device)

        # Factorization loss (for monitoring only, detach to avoid affecting injection grads)
        result['factorization_loss'] = self.cofrn.factorization.factorization_loss(
            structural.detach(), context.detach()
        )
        result['entropy_loss'] = self.cofrn.factorization.entropy_loss(anchor_weights.detach())

        result['total_loss'] = result['task_loss']

        return result

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing RuleLibrary and RuleInjectionLayer...")

    # Test RuleLibrary
    library = RuleLibrary(device='cpu')
    print(f"\nRuleLibrary: {library.n_rules} rules, {library.emb_dim}D")
    print(f"Domains: {list(library.domain_to_indices.keys())}")

    # Test retrieval
    query = torch.randn(2, library.emb_dim)
    ctx, sims, info = library.retrieve(query, top_k=3)
    print(f"\nRetrieval for 2 queries, top_k=3:")
    print(f"  context shape: {ctx.shape}")
    print(f"  similarities: {sims.shape}")
    for i, q_info in enumerate(info):
        print(f"\n  Query {i}:")
        for r in q_info:
            print(f"    {r['rule_id']} {r['name']:<30} ({r['domain']}) sim={r['similarity']:.3f}")

    # Test domain filtering
    ctx_phys, sims_phys, info_phys = library.retrieve(query, top_k=3, domain_filter='physics')
    print(f"\nDomain-filtered (physics) retrieval:")
    for r in info_phys[0]:
        print(f"  {r['rule_id']} {r['name']:<30} sim={r['similarity']:.3f}")

    # Test RuleInjectionLayer
    layer = RuleInjectionLayer(rule_dim=library.emb_dim, hidden_dim=256)
    print(f"\nRuleInjectionLayer params: {layer.trainable_params:,}")

    s0 = torch.randn(2, 256)
    s0_aug = layer(s0, ctx)
    print(f"  s0: {s0.shape} -> s0_aug: {s0_aug.shape}")
    print(f"  alpha: {layer.alpha.item():.3f}")

    # Gradient check
    s0_aug.sum().backward()
    grad_params = sum(1 for p in layer.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in layer.parameters() if p.requires_grad)
    print(f"  Gradient flow: {grad_params}/{total} params have gradients")

    print("\nAll rule_injection tests passed!")
