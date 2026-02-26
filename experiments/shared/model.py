#!/usr/bin/env python3
"""
Full CO-FRN Model Assembly (Section 3 of draft).

Chains three modules:
1. Semantic Encoder (frozen GPT-2 + trainable projection)
2. Continuous Operator Manifold Factorization
3. Reasoning Engine (HilbertTree for multi-step, DirectScorer for single-step)

Adapted from Paper 11 E3 models.py::HilbertTreeModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass, field

from .encoder import FrozenEncoder, PrecomputedEncoder
from .factorization import ContinuousFactorizationModule
from .reasoning_engine import ReasoningEngine, ARTIRoutedScorer
from .operators import SimpleMLPOperator


@dataclass
class COFRNConfig:
    """Configuration for the CO-FRN model."""
    # Encoder
    encoder_model: str = 'gpt2'
    hidden_dim: int = 256
    use_precomputed: bool = False  # if True, skip transformer, use cached embeddings
    encoder_input_dim: int = 768  # for PrecomputedEncoder

    # Factorization
    struct_dim: int = 128
    context_dim: int = 128

    # Operator manifold
    manifold_dim: int = 10
    n_anchors: int = 16
    rank: int = 16
    tau_init: float = 1.0
    tau_decay: float = 0.95
    tau_floor: float = 0.1

    # Reasoning engine
    task_type: str = 'single_step'  # 'single_step' or 'multi_step'
    beam_width: int = 2
    max_depth: int = 3
    delta: float = 0.8

    # Ablation flags (E2b component ablation)
    disable_manifold: bool = False   # Bypass manifold: use s0 directly as transformed
    disable_tree: bool = False       # Force single_step (forward_direct) even for multi_step
    use_mlp_operator: bool = False   # Use SimpleMLPOperator as tree operator instead of manifold

    # E10 fix: cosine scoring (replaces MLP scorers that collapsed to answer-only)
    use_cosine_scoring: bool = False

    # E15: ARTI-routed scoring (blends cosine + direct classification)
    use_routed_scoring: bool = False
    arti_checkpoint: str = ''  # path to pretrained ARTI v1 checkpoint
    arti_encoder_dim: int = 384  # ARTI expected input dim

    # E13: Selective encoder unfreezing
    unfreeze_encoder_layers: int = 0  # 0=fully frozen, 2/6/12=unfreeze last N blocks

    # E18: Freeze answer encoding path while unfreezing question encoding
    freeze_answer_encoder: bool = False

    # Training
    lambda_factorize: float = 0.1
    lambda_coherence: float = 0.01
    lambda_entropy: float = 0.01

    # Entropy schedule (E5: operator specialization)
    entropy_schedule: str = 'constant'  # 'constant', 'disabled', 'anneal'
    entropy_anneal_epochs: int = 10


class COFRN(nn.Module):
    """
    Continuous Operator Factorized Reasoning Network.

    Full architecture:
        text -> FrozenEncoder -> |s0>
             -> ContinuousFactorizationModule -> (O(s0), |o>, |d>, weights)
             -> ReasoningEngine -> answer scores

    Parameter budget (~726K for GPT-2 124M, default config):
        Encoder projection: ~200K
        Structural MLP: ~130K
        Context MLP: ~130K
        MI discriminator: ~66K
        Operator manifold: ~131K
        Manifold projection: ~2.6K
        Reasoning/scoring: ~66K
    """

    def __init__(self, config: COFRNConfig):
        super().__init__()
        self.config = config
        self._current_epoch = 0  # tracked for entropy schedule

        # Module 1: Semantic Encoder
        if config.use_precomputed:
            self.encoder = PrecomputedEncoder(
                input_dim=config.encoder_input_dim,
                hidden_dim=config.hidden_dim,
            )
        else:
            self.encoder = FrozenEncoder(
                model_name=config.encoder_model,
                hidden_dim=config.hidden_dim,
                load_pretrained=True,
                unfreeze_last_n=config.unfreeze_encoder_layers,
            )

        # Module 2: Continuous Operator Manifold Factorization
        self.factorization = ContinuousFactorizationModule(
            hidden_dim=config.hidden_dim,
            struct_dim=config.struct_dim,
            context_dim=config.context_dim,
            manifold_dim=config.manifold_dim,
            n_anchors=config.n_anchors,
            rank=config.rank,
            tau_init=config.tau_init,
            tau_decay=config.tau_decay,
            tau_floor=config.tau_floor,
        )

        # Module 3: Reasoning Engine
        self.reasoning = ReasoningEngine(
            hidden_dim=config.hidden_dim,
            beam_width=config.beam_width,
            max_depth=config.max_depth,
            delta=config.delta,
            use_cosine_scoring=config.use_cosine_scoring,
        )

        # Optional MLP operator for tree-only ablation (E2b)
        self.mlp_operator = None
        if config.use_mlp_operator:
            self.mlp_operator = SimpleMLPOperator(
                hidden_dim=config.hidden_dim,
                struct_dim=config.struct_dim,
            )

        # E15: ARTI-routed scoring
        self.arti = None
        self.arti_adapter = None
        self.routed_scorer = None
        if config.use_routed_scoring:
            from .arti import ARTI, ARTIConfig
            arti_config = ARTIConfig(encoder_dim=config.arti_encoder_dim)
            self.arti = ARTI(arti_config)
            if config.arti_checkpoint:
                state = torch.load(config.arti_checkpoint, weights_only=True, map_location='cpu')
                self.arti.load_state_dict(state)
            # Freeze ARTI — we only use it for routing
            for p in self.arti.parameters():
                p.requires_grad = False
            self.arti.eval()
            # Adapter: project s0 (hidden_dim) to ARTI's expected dim
            self.arti_adapter = nn.Linear(config.hidden_dim, config.arti_encoder_dim)
            # Routed scorer
            self.routed_scorer = ARTIRoutedScorer(
                hidden_dim=config.hidden_dim,
                max_choices=4,
                n_arti_types=10,
            )

        # Answer encoder: project answer text embeddings to hidden_dim
        # (for scoring against transformed state)
        # Input dim matches encoder_input_dim in precomputed mode (answers stored
        # at encoder dim, which may differ from model's hidden_dim at non-default budgets)
        _answer_in = config.encoder_input_dim if config.use_precomputed else config.hidden_dim
        self.answer_proj = nn.Sequential(
            nn.Linear(_answer_in, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def encode_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input text to |s0>."""
        if isinstance(self.encoder, PrecomputedEncoder):
            return self.encoder(embeddings)
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            embeddings=embeddings,
        )

    def encode_answers(
        self,
        answer_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode answer choices from pre-computed embeddings.

        Args:
            answer_embeddings: [B, n_answers, hidden_dim]

        Returns:
            projected: [B, n_answers, hidden_dim]
        """
        B, N, D = answer_embeddings.shape
        flat = answer_embeddings.reshape(B * N, D)
        projected = self.answer_proj(flat)
        projected = F.normalize(projected, dim=-1)
        return projected.reshape(B, N, -1)

    def encode_answers_from_text(
        self,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode answer choices from raw text tokens (for unfrozen encoder).

        Routes through the full encoder (transformer + projection) to get
        hidden_dim vectors, then through answer_proj.

        Args:
            answer_input_ids: [B, n_answers, L] token IDs
            answer_attention_mask: [B, n_answers, L] attention masks

        Returns:
            projected: [B, n_answers, hidden_dim]
        """
        B, N, L = answer_input_ids.shape
        flat_ids = answer_input_ids.reshape(B * N, L)
        flat_mask = answer_attention_mask.reshape(B * N, L)

        # E18: Freeze transformer for answer encoding — prevents answer embedding
        # destabilization while question path remains unfrozen
        if self.config.freeze_answer_encoder:
            with torch.no_grad():
                s0 = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
            s0 = s0.detach()  # Cut gradient flow to transformer
        else:
            # Full encoder: transformer → mean-pool → projection → [B*N, hidden_dim]
            s0 = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        # Project through answer projection
        projected = self.answer_proj(s0)
        projected = F.normalize(projected, dim=-1)
        return projected.reshape(B, N, -1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        answer_embeddings: Optional[torch.Tensor] = None,
        answer_input_ids: Optional[torch.Tensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        evidence_embeddings: Optional[torch.Tensor] = None,
        depth: Optional[int] = None,
        valid_choices: Optional[torch.Tensor] = None,
        st_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            input_ids: [B, L] token IDs (for FrozenEncoder)
            attention_mask: [B, L] mask
            embeddings: [B, D_model] pre-extracted embeddings
            answer_embeddings: [B, n_answers, hidden_dim] answer encodings
            answer_input_ids: [B, n_answers, L] answer token IDs (for unfrozen encoder)
            answer_attention_mask: [B, n_answers, L] answer masks (for unfrozen encoder)
            labels: [B] correct answer indices (for loss computation)
            evidence_embeddings: [B, hidden_dim] evidence for multi-step
            depth: tree depth for multi-step
            valid_choices: [B, n_answers] bool mask for valid answer slots
            st_embeddings: [B, 384] precomputed MiniLM embeddings for ARTI (E17).
                When provided, ARTI uses these directly instead of adapter(s0).

        Returns:
            Dict with scores, losses, diagnostics
        """
        # Module 1: Encode
        s0 = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            embeddings=embeddings,
        )

        # Module 2: Factorize and apply operator
        fact_out = self.factorization(s0)
        structural = fact_out['structural']
        context = fact_out['context']
        anchor_weights = fact_out['weights']

        # Ablation: disable_manifold bypasses the manifold transform
        if self.config.disable_manifold:
            transformed = s0
        else:
            transformed = fact_out['transformed']

        # Module 3: Reasoning
        if answer_input_ids is not None and answer_attention_mask is not None:
            answer_enc = self.encode_answers_from_text(
                answer_input_ids, answer_attention_mask
            )
        elif answer_embeddings is not None:
            answer_enc = self.encode_answers(answer_embeddings)
        else:
            answer_enc = None

        # Determine effective task type (disable_tree forces single_step)
        effective_task_type = self.config.task_type
        if self.config.disable_tree:
            effective_task_type = 'single_step'

        # Select operator_fn for tree: manifold (default) or MLP (ablation)
        if self.mlp_operator is not None:
            operator_fn = self.mlp_operator.forward
        else:
            operator_fn = self.factorization.manifold.forward

        # E15: ARTI-routed scoring — blends cosine + direct via ARTI type probs
        if self.routed_scorer is not None and answer_enc is not None:
            # Get ARTI type probabilities
            # E17: when st_embeddings provided, feed native MiniLM to ARTI (no adapter)
            # E15 fallback: project s0 through adapter
            with torch.no_grad():
                if st_embeddings is not None:
                    arti_out = self.arti(st_embeddings)
                else:
                    arti_input = self.arti_adapter(s0)
                    arti_out = self.arti(arti_input)
                arti_probs = arti_out['probabilities']  # [B, 10]

            # Run tree/manifold to get transformed state (same as non-routed)
            if effective_task_type == 'multi_step':
                evidence = evidence_embeddings if evidence_embeddings is not None else transformed
                reason_out = self.reasoning.forward_multistep(
                    transformed=transformed,
                    evidence=evidence,
                    operator_fn=operator_fn,
                    structural=structural,
                    answer_encodings=answer_enc,
                    depth=depth,
                )
                # Use best hypothesis as input to routed scorer
                h_weights = F.softmax(reason_out['hypothesis_scores'], dim=-1)
                best_hyp = (h_weights.unsqueeze(-1) * reason_out['hypothesis_states']).sum(dim=1)
                scorer_input = best_hyp
            else:
                reason_out = {'coherence_loss': torch.tensor(0.0, device=s0.device)}
                scorer_input = transformed

            # Score via routed scorer
            scores, router_alpha = self.routed_scorer(
                scorer_input, answer_enc, arti_probs, valid_choices
            )
            reason_out['scores'] = scores
            reason_out['router_alpha'] = router_alpha

        elif effective_task_type == 'multi_step' and answer_enc is not None:
            evidence = evidence_embeddings if evidence_embeddings is not None else transformed
            reason_out = self.reasoning.forward_multistep(
                transformed=transformed,
                evidence=evidence,
                operator_fn=operator_fn,
                structural=structural,
                answer_encodings=answer_enc,
                depth=depth,
            )
        elif answer_enc is not None:
            reason_out = self.reasoning.forward_direct(
                transformed=transformed,
                answer_encodings=answer_enc,
            )
        else:
            reason_out = {
                'scores': None,
                'coherence_loss': torch.tensor(0.0, device=s0.device),
            }

        # Compute losses
        result = {
            'transformed': transformed,
            'structural': structural,
            'context': context,
            'anchor_weights': anchor_weights,
            'scores': reason_out.get('scores'),
            'coherence_loss': reason_out.get('coherence_loss', torch.tensor(0.0, device=s0.device)),
        }

        # Router alpha diagnostic (E15)
        if 'router_alpha' in reason_out:
            result['router_alpha'] = reason_out['router_alpha']

        # Task loss
        if labels is not None and reason_out.get('scores') is not None:
            result['task_loss'] = F.cross_entropy(
                reason_out['scores'], labels
            )
            result['predicted'] = reason_out['scores'].argmax(dim=-1)
            result['correct'] = (result['predicted'] == labels).float()
        else:
            result['task_loss'] = torch.tensor(0.0, device=s0.device)

        # Factorization loss (MI discriminator)
        result['factorization_loss'] = self.factorization.factorization_loss(
            structural, context
        )

        # Entropy regularizer (anchor utilization)
        result['entropy_loss'] = self.factorization.entropy_loss(anchor_weights)

        # Compute effective lambda_entropy based on schedule
        lambda_e = self._get_effective_lambda_entropy()

        # Total loss (default weighting from config; the Trainer overrides
        # this with its own annealed weights via _forward_step)
        result['total_loss'] = (
            result['task_loss']
            + self.config.lambda_factorize * result['factorization_loss']
            + self.config.lambda_coherence * result['coherence_loss']
            + lambda_e * result['entropy_loss']
        )

        # Tree outputs (if multi-step)
        if 'hypothesis_states' in reason_out:
            result['hypothesis_states'] = reason_out['hypothesis_states']
            result['hypothesis_scores'] = reason_out['hypothesis_scores']

        return result

    def _get_effective_lambda_entropy(self) -> float:
        """Compute effective lambda_entropy based on entropy_schedule config."""
        schedule = self.config.entropy_schedule
        if schedule == 'disabled':
            return 0.0
        elif schedule == 'anneal':
            anneal_epochs = max(1, self.config.entropy_anneal_epochs)
            return self.config.lambda_entropy * max(0.0, 1.0 - self._current_epoch / anneal_epochs)
        else:  # 'constant' (default)
            return self.config.lambda_entropy

    def anneal_temperature(self, epoch: int):
        """Anneal manifold temperature. Call once per epoch."""
        self._current_epoch = epoch
        self.factorization.anneal_temperature(epoch)

    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters only."""
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    @property
    def total_params(self) -> int:
        """Total parameters including frozen."""
        return sum(p.numel() for p in self.parameters())

    def get_param_breakdown(self) -> Dict[str, int]:
        """Detailed parameter count by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        breakdown = {
            'encoder_projection': count(self.encoder),
            'factorization_struct_mlp': count(self.factorization.factorizer.struct_mlp),
            'factorization_context_mlp': count(self.factorization.factorizer.context_mlp),
            'factorization_mi_disc': count(self.factorization.factorizer.mi_discriminator),
            'manifold_anchors': self.factorization.manifold.anchors.numel(),
            'manifold_operators': (
                self.factorization.manifold.U_all.numel()
                + self.factorization.manifold.V_all.numel()
            ),
            'manifold_projection': count(self.factorization.manifold.manifold_proj),
            'reasoning_engine': count(self.reasoning),
            'answer_projection': count(self.answer_proj),
        }
        if self.mlp_operator is not None:
            breakdown['mlp_operator'] = count(self.mlp_operator)
        if self.routed_scorer is not None:
            breakdown['arti_adapter'] = count(self.arti_adapter)
            breakdown['routed_scorer'] = count(self.routed_scorer)
        breakdown['total_trainable'] = self.trainable_params
        return breakdown


if __name__ == "__main__":
    print("Testing CO-FRN model...")

    # Test with precomputed embeddings (no transformer download)
    config = COFRNConfig(
        encoder_model='gpt2',
        hidden_dim=256,
        use_precomputed=True,
        encoder_input_dim=768,
        struct_dim=128,
        context_dim=128,
        manifold_dim=10,
        n_anchors=16,
        rank=16,
        task_type='single_step',
    )

    model = COFRN(config)

    breakdown = model.get_param_breakdown()
    print("\nParameter breakdown:")
    for name, count in breakdown.items():
        print(f"  {name}: {count:,}")

    # Forward pass
    batch_size = 4
    n_answers = 4
    embeddings = torch.randn(batch_size, 768)
    answer_emb = torch.randn(batch_size, n_answers, 256)
    labels = torch.randint(0, n_answers, (batch_size,))

    output = model(
        embeddings=embeddings,
        answer_embeddings=answer_emb,
        labels=labels,
    )

    print(f"\nForward pass (single-step):")
    print(f"  scores: {output['scores'].shape}")
    print(f"  task_loss: {output['task_loss'].item():.4f}")
    print(f"  factorization_loss: {output['factorization_loss'].item():.4f}")
    print(f"  entropy_loss: {output['entropy_loss'].item():.4f}")
    print(f"  coherence_loss: {output['coherence_loss'].item():.4f}")
    print(f"  total_loss: {output['total_loss'].item():.4f}")
    print(f"  predicted: {output['predicted'].tolist()}")

    # Backward pass
    output['total_loss'].backward()
    grad_params = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\nGradient flow: {grad_params}/{total} trainable params have gradients")

    # Test multi-step config
    print("\n--- Multi-step mode ---")
    config_ms = COFRNConfig(
        use_precomputed=True,
        encoder_input_dim=768,
        hidden_dim=256,
        task_type='multi_step',
        beam_width=2,
        max_depth=2,
    )
    model_ms = COFRN(config_ms)

    output_ms = model_ms(
        embeddings=embeddings,
        answer_embeddings=answer_emb,
        labels=labels,
    )
    print(f"  scores: {output_ms['scores'].shape}")
    print(f"  coherence_loss: {output_ms['coherence_loss'].item():.4f}")
    print(f"  total_loss: {output_ms['total_loss'].item():.4f}")

    print("\nCO-FRN model tests passed!")
