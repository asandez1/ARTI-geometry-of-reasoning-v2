#!/usr/bin/env python3
"""
GeometricReasoningController: Bridges ARTI type detection to dynamic
reasoning engine configuration.

Uses the detected reasoning type to:
  1. Route between fast path (DirectScorer) and structured path (HilbertTreeFast)
  2. Select type-specific tree depth and coherence bounds
  3. Soft-mask operator manifold anchors (top-k focused)
  4. Blend fast/structured outputs during training (hard routing at inference)

The TypeClassifier is an MLP (~16,774 params) on the 256D s0 embeddings
(projected encoder output, before factorization) — retains full semantic signal.

Six core types (merged from ARTI's 10):
  0: CauseEffect    (PhysCause + BehvCause + SysCause)
  1: Deduction       (Deduction)
  2: Induction       (Induction)
  3: Analogy         (Analogy)
  4: Conservation    (Conservation)
  5: Counterfactual  (Counterfactual + Abduction + Decomposition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .reasoning_types import ReasoningType


# ─── Type Merge Map ───────────────────────────────────────────────────────────

# Maps original 10 ARTI type IDs -> 6 core type IDs
TYPE_MERGE_MAP: Dict[int, int] = {
    int(ReasoningType.PHYSICAL_CAUSE):  0,   # CauseEffect
    int(ReasoningType.BEHAVIORAL_CAUSE): 0,  # CauseEffect
    int(ReasoningType.SYSTEMIC_CAUSE):  0,   # CauseEffect
    int(ReasoningType.DEDUCTION):       1,   # Deduction
    int(ReasoningType.INDUCTION):       2,   # Induction
    int(ReasoningType.ANALOGY):         3,   # Analogy
    int(ReasoningType.CONSERVATION):    4,   # Conservation
    int(ReasoningType.COUNTERFACTUAL):  5,   # Counterfactual
    int(ReasoningType.ABDUCTION):       5,   # Counterfactual
    int(ReasoningType.DECOMPOSITION):   5,   # Counterfactual
}

CORE_TYPE_NAMES: List[str] = [
    "CauseEffect",
    "Deduction",
    "Induction",
    "Analogy",
    "Conservation",
    "Counterfactual",
]

N_CORE_TYPES = len(CORE_TYPE_NAMES)


def merge_labels(labels_10: torch.Tensor) -> torch.Tensor:
    """Convert 10-type ARTI labels to 6 core type labels."""
    mapping = torch.zeros(10, dtype=torch.long, device=labels_10.device)
    for src, dst in TYPE_MERGE_MAP.items():
        mapping[src] = dst
    return mapping[labels_10]


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    """Configuration for GeometricReasoningController."""
    n_core_types: int = 6
    confidence_threshold: float = 0.65

    # Which types trigger structured (tree) vs fast (direct) path
    structured_types: List[int] = field(
        default_factory=lambda: [2, 3, 5]  # Induction, Analogy, Counterfactual
    )
    fast_types: List[int] = field(
        default_factory=lambda: [0, 1, 4]  # CauseEffect, Deduction, Conservation
    )

    # Per-type tree depth
    type_depths: Dict[int, int] = field(default_factory=lambda: {
        0: 1,  # CauseEffect: direct causal chain
        1: 1,  # Deduction: single logical step
        2: 3,  # Induction: pattern accumulation
        3: 2,  # Analogy: cross-domain mapping
        4: 2,  # Conservation: invariance check
        5: 3,  # Counterfactual: branching hypothesis
    })

    # Per-type coherence bound (delta for HilbertTree)
    type_deltas: Dict[int, float] = field(default_factory=lambda: {
        0: 0.8,  # CauseEffect
        1: 0.9,  # Deduction: strict
        2: 0.6,  # Induction: loose for pattern search
        3: 0.6,  # Analogy: loose for cross-domain
        4: 0.9,  # Conservation: strict invariance
        5: 0.7,  # Counterfactual
    })

    # Anchor masking
    top_k_anchors: int = 5
    mask_temperature: float = 0.3

    # Dimensions (must match COFRN config)
    hidden_dim: int = 256
    manifold_dim: int = 10
    struct_dim: int = 128


# ─── TypeClassifier ───────────────────────────────────────────────────────────

class TypeClassifier(nn.Module):
    """
    MLP that classifies reasoning type from s0 embeddings.

    Uses 256D s0 (projected encoder output, before factorization) which
    retains the full semantic signal needed for type discrimination.

    Architecture: Linear(256, 64) -> GELU -> Linear(64, 6)
    Parameters:  256*64 + 64 + 64*6 + 6 = 16,774 params
    """

    def __init__(self, input_dim: int = 256, n_classes: int = 6, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify reasoning type from structural features.

        Args:
            features: [B, input_dim] structural features from factorization

        Returns:
            type_probs: [B, n_classes] softmax probabilities
            detected_type: [B] argmax type index
            confidence: [B] max probability
        """
        logits = self.mlp(features)  # [B, n_classes]
        type_probs = F.softmax(logits, dim=-1)
        confidence, detected_type = type_probs.max(dim=-1)
        return type_probs, detected_type, confidence

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── GeometricReasoningController ─────────────────────────────────────────────

class GeometricReasoningController(nn.Module):
    """
    Bridges reasoning type detection to dynamic engine configuration.

    Architecture:
        s0 -> ContinuousFactorizationModule (reused, frozen or shared)
           -> structural -> manifold_proj -> manifold_coords [B, 10]
           -> TypeClassifier -> type_probs [B, 6], detected_type, confidence
           -> Router -> routing_weight in [0, 1]
           -> Blend of fast path (DirectScorer) and structured path (HilbertTreeFast)

    During training: soft blend via routing_weight (differentiable)
    During inference: hard routing (fast or structured, no blend)
    """

    def __init__(
        self,
        config: ControllerConfig,
        factorization: nn.Module,
        reasoning_engine: nn.Module,
    ):
        """
        Args:
            config: ControllerConfig
            factorization: ContinuousFactorizationModule (reused from COFRN)
            reasoning_engine: ReasoningEngine (reused from COFRN)
        """
        super().__init__()
        self.config = config
        self.factorization = factorization
        self.reasoning = reasoning_engine

        # TypeClassifier on s0 (256D projected embeddings, before factorization)
        # s0 retains the full semantic signal needed for type discrimination
        self.type_clf = TypeClassifier(
            input_dim=config.hidden_dim,
            n_classes=config.n_core_types,
        )

        # Route gate: type_probs(6) + entropy(1) + confidence(1) -> scalar
        self.route_gate = nn.Sequential(
            nn.Linear(config.n_core_types + 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Answer projection (same as COFRN)
        self.answer_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def detect_type(
        self,
        s0: torch.Tensor,
        structural: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect reasoning type from s0 (pre-factorization embeddings).

        Uses 256D s0 (projected encoder output before factorization) because
        it retains the full semantic signal needed for type discrimination.
        Post-factorization features (structural, manifold_coords) are shaped
        by MI discriminator for QA performance, not reasoning type separation.

        Args:
            s0: [B, hidden_dim] projected encoder output (before factorization)
            structural: [B, struct_dim] structural component (for manifold coords)

        Returns:
            type_probs: [B, n_core_types]
            detected_type: [B]
            confidence: [B]
            manifold_coords: [B, manifold_dim]
        """
        # Classify from s0 (pre-factorization, full semantic signal)
        type_probs, detected_type, confidence = self.type_clf(s0)

        # Also compute manifold coords for diagnostics/anchor masking
        manifold_coords = self.factorization.manifold.manifold_proj(structural)

        return type_probs, detected_type, confidence, manifold_coords

    def compute_routing(
        self,
        type_probs: torch.Tensor,
        confidence: torch.Tensor,
        anchor_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute routing weight: 0 = fast path, 1 = structured path.

        Uses a learned gate on type_probs + distribution entropy + confidence.

        Args:
            type_probs: [B, n_core_types]
            confidence: [B]
            anchor_weights: [B, n_anchors] (not directly used but available)

        Returns:
            routing_weight: [B, 1] in [0, 1] (sigmoid)
        """
        # Entropy of type distribution
        entropy = -(type_probs * torch.log(type_probs + 1e-10)).sum(dim=-1, keepdim=True)
        # [B, 1]

        # Concatenate: type_probs(6) + entropy(1) + confidence(1)
        gate_input = torch.cat([
            type_probs,
            entropy,
            confidence.unsqueeze(-1),
        ], dim=-1)  # [B, 8]

        routing_weight = torch.sigmoid(self.route_gate(gate_input))  # [B, 1]
        return routing_weight

    def create_soft_masked_operator_fn(
        self,
        anchor_weights: torch.Tensor,
        top_k: int,
        temperature: float,
    ):
        """
        Create a masked operator function that focuses on top-k anchors.

        During training: soft mask (differentiable via temperature-scaled softmax)
        During inference: hard top-k mask with renormalization

        Args:
            anchor_weights: [B, n_anchors] original attention weights
            top_k: number of anchors to keep
            temperature: masking sharpness

        Returns:
            callable(x, structural) -> (Ox, weights) with masked weights
        """
        manifold = self.factorization.manifold

        def masked_operator_fn(x: torch.Tensor, structural: torch.Tensor):
            # Compute fresh attention weights
            weights = manifold.compute_attention_weights(structural)

            # Soft mask: sharpen existing weights via temperature-scaled softmax
            # This keeps gradient flow through all anchors but concentrates mass
            masked_weights = F.softmax(
                torch.log(weights + 1e-10) / temperature,
                dim=-1,
            )

            # Apply operator with masked weights
            Ox = manifold.apply_operator(x, masked_weights)
            return Ox, masked_weights

        return masked_operator_fn

    def get_type_config(
        self,
        detected_type: torch.Tensor,
    ) -> Tuple[int, float]:
        """
        Get type-specific depth and delta for a single example.

        Args:
            detected_type: scalar tensor or int

        Returns:
            (depth, delta) tuple
        """
        t = int(detected_type.item()) if isinstance(detected_type, torch.Tensor) else int(detected_type)
        depth = self.config.type_depths.get(t, 2)
        delta = self.config.type_deltas.get(t, 0.8)
        return depth, delta

    def encode_answers(
        self,
        answer_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project answer embeddings.

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

    def forward(
        self,
        s0: torch.Tensor,
        answer_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        evidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with soft blending (training mode).

        Runs both fast and structured paths, blends by routing_weight.

        Args:
            s0: [B, hidden_dim] encoder output (already projected)
            answer_embeddings: [B, n_answers, hidden_dim]
            labels: [B] correct answer indices
            evidence: [B, hidden_dim] evidence for tree (defaults to transformed)

        Returns:
            Dict with scores, losses, type info, routing diagnostics
        """
        device = s0.device
        B = s0.shape[0]

        # Step 1: Factorize
        fact_out = self.factorization(s0)
        structural = fact_out['structural']
        context = fact_out['context']
        transformed = fact_out['transformed']
        anchor_weights = fact_out['weights']

        # Step 2: Detect reasoning type (from s0, before factorization)
        type_probs, detected_type, confidence, manifold_coords = self.detect_type(s0, structural)

        # Step 3: Compute routing weight
        routing_weight = self.compute_routing(type_probs, confidence, anchor_weights)

        # Step 4: Encode answers
        answer_enc = None
        if answer_embeddings is not None:
            answer_enc = self.encode_answers(answer_embeddings)

        # Step 5: Run both paths
        # Fast path (direct scoring)
        if answer_enc is not None:
            fast_out = self.reasoning.forward_direct(transformed, answer_enc)
            fast_scores = fast_out['scores']  # [B, n_answers]
        else:
            fast_scores = None

        # Structured path (Hilbert tree with type-specific config)
        if answer_enc is not None:
            # Use majority detected type for batch depth/delta
            mode_type = detected_type.mode().values.item() if B > 1 else detected_type[0].item()
            depth, delta = self.get_type_config(mode_type)

            # Create masked operator function
            masked_op_fn = self.create_soft_masked_operator_fn(
                anchor_weights,
                top_k=self.config.top_k_anchors,
                temperature=self.config.mask_temperature,
            )

            # Temporarily set tree delta
            old_delta = self.reasoning.tree.delta
            self.reasoning.tree.delta = delta

            evidence_vec = evidence if evidence is not None else transformed
            struct_out = self.reasoning.forward_multistep(
                transformed=transformed,
                evidence=evidence_vec,
                operator_fn=masked_op_fn,
                structural=structural,
                answer_encodings=answer_enc,
                depth=depth,
            )
            struct_scores = struct_out['scores']  # [B, n_answers]
            coherence_loss = struct_out['coherence_loss']

            # Restore tree delta
            self.reasoning.tree.delta = old_delta
        else:
            struct_scores = None
            coherence_loss = torch.tensor(0.0, device=device)

        # Step 6: Blend scores
        if fast_scores is not None and struct_scores is not None:
            # routing_weight: [B, 1], 0=fast, 1=structured
            scores = (1 - routing_weight) * fast_scores + routing_weight * struct_scores
        elif fast_scores is not None:
            scores = fast_scores
        else:
            scores = None

        # Build result
        result = {
            'scores': scores,
            'transformed': transformed,
            'structural': structural,
            'context': context,
            'anchor_weights': anchor_weights,
            'type_probs': type_probs,
            'detected_type': detected_type,
            'confidence': confidence,
            'manifold_coords': manifold_coords,
            'routing_weight': routing_weight,
            'coherence_loss': coherence_loss,
        }

        # Task loss
        if labels is not None and scores is not None:
            result['task_loss'] = F.cross_entropy(scores, labels)
            result['predicted'] = scores.argmax(dim=-1)
            result['correct'] = (result['predicted'] == labels).float()
        else:
            result['task_loss'] = torch.tensor(0.0, device=device)

        # Factorization loss
        result['factorization_loss'] = self.factorization.factorization_loss(
            structural, context
        )

        # Entropy loss
        result['entropy_loss'] = self.factorization.entropy_loss(anchor_weights)

        # Total loss
        result['total_loss'] = (
            result['task_loss']
            + 0.1 * result['factorization_loss']
            + 0.01 * result['coherence_loss']
            + 0.01 * result['entropy_loss']
        )

        return result

    def forward_inference(
        self,
        s0: torch.Tensor,
        answer_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-only forward with hard routing (no blending).

        Routes each example to either fast or structured path based on
        detected type and confidence threshold.

        Args:
            s0: [B, hidden_dim]
            answer_embeddings: [B, n_answers, hidden_dim]

        Returns:
            Dict with scores, type info, routing decisions
        """
        device = s0.device
        B = s0.shape[0]

        # Factorize
        fact_out = self.factorization(s0)
        structural = fact_out['structural']
        transformed = fact_out['transformed']
        anchor_weights = fact_out['weights']

        # Detect type (from s0, before factorization)
        type_probs, detected_type, confidence, manifold_coords = self.detect_type(s0, structural)

        # Encode answers
        answer_enc = self.encode_answers(answer_embeddings)

        # Hard routing per example
        structured_set = set(self.config.structured_types)
        all_scores = torch.zeros(B, answer_enc.shape[1], device=device)
        route_decisions = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            t = detected_type[i].item()
            conf = confidence[i].item()

            # Route to structured if: type is structured OR confidence is low
            use_structured = (t in structured_set) or (conf < self.config.confidence_threshold)

            if use_structured:
                route_decisions[i] = 1
                depth, delta = self.get_type_config(t)

                # Hard top-k masking for inference
                w = anchor_weights[i:i+1]  # [1, n_anchors]
                topk_vals, topk_idx = w.topk(self.config.top_k_anchors, dim=-1)
                mask = torch.zeros_like(w)
                mask.scatter_(1, topk_idx, 1.0)
                masked_w = w * mask
                masked_w = masked_w / (masked_w.sum(dim=-1, keepdim=True) + 1e-10)

                def hard_masked_op(x, s, _mw=masked_w):
                    Ox = self.factorization.manifold.apply_operator(x, _mw.expand(x.shape[0], -1))
                    return Ox, _mw

                old_delta = self.reasoning.tree.delta
                self.reasoning.tree.delta = delta

                out = self.reasoning.forward_multistep(
                    transformed=transformed[i:i+1],
                    evidence=transformed[i:i+1],
                    operator_fn=hard_masked_op,
                    structural=structural[i:i+1],
                    answer_encodings=answer_enc[i:i+1],
                    depth=depth,
                )
                all_scores[i] = out['scores'].squeeze(0)
                self.reasoning.tree.delta = old_delta
            else:
                route_decisions[i] = 0
                out = self.reasoning.forward_direct(
                    transformed=transformed[i:i+1],
                    answer_encodings=answer_enc[i:i+1],
                )
                all_scores[i] = out['scores'].squeeze(0)

        return {
            'scores': all_scores,
            'predicted': all_scores.argmax(dim=-1),
            'type_probs': type_probs,
            'detected_type': detected_type,
            'confidence': confidence,
            'manifold_coords': manifold_coords,
            'route_decisions': route_decisions,
            'anchor_weights': anchor_weights,
        }

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_diagnostics(
        self,
        result: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Extract human-readable diagnostics from a forward pass result.

        Args:
            result: output dict from forward() or forward_inference()

        Returns:
            Dict with diagnostic metrics
        """
        diag = {}

        # Type distribution
        type_probs = result['type_probs']
        detected = result['detected_type']
        confidence = result['confidence']
        B = type_probs.shape[0]

        diag['mean_confidence'] = confidence.mean().item()
        diag['min_confidence'] = confidence.min().item()

        # Type distribution entropy
        ent = -(type_probs * torch.log(type_probs + 1e-10)).sum(dim=-1)
        diag['mean_type_entropy'] = ent.mean().item()

        # Routing stats
        if 'routing_weight' in result:
            rw = result['routing_weight']
            diag['mean_routing_weight'] = rw.mean().item()
            diag['pct_structured'] = (rw > 0.5).float().mean().item()
        elif 'route_decisions' in result:
            rd = result['route_decisions']
            diag['pct_structured'] = rd.float().mean().item()

        # Anchor weights stats
        if 'anchor_weights' in result:
            aw = result['anchor_weights']
            mean_aw = aw.mean(dim=0)
            pr = (mean_aw.sum() ** 2) / ((mean_aw ** 2).sum() + 1e-10)
            diag['anchor_participation_ratio'] = pr.item()

        # Per-type counts
        for t in range(self.config.n_core_types):
            count = (detected == t).sum().item()
            diag[f'type_{CORE_TYPE_NAMES[t]}_count'] = count
            diag[f'type_{CORE_TYPE_NAMES[t]}_pct'] = count / max(B, 1)

        return diag


if __name__ == "__main__":
    print("Testing GeometricReasoningController...")

    from .factorization import ContinuousFactorizationModule
    from .reasoning_engine import ReasoningEngine

    # Create components
    hidden_dim = 256
    struct_dim = 128
    manifold_dim = 10
    n_anchors = 16
    batch_size = 4
    n_answers = 4

    factorization = ContinuousFactorizationModule(
        hidden_dim=hidden_dim,
        struct_dim=struct_dim,
        context_dim=128,
        manifold_dim=manifold_dim,
        n_anchors=n_anchors,
    )

    reasoning = ReasoningEngine(
        hidden_dim=hidden_dim,
        beam_width=2,
        max_depth=3,
    )

    config = ControllerConfig(
        hidden_dim=hidden_dim,
        manifold_dim=manifold_dim,
        struct_dim=struct_dim,
    )

    controller = GeometricReasoningController(config, factorization, reasoning)

    # Count params
    total = sum(p.numel() for p in controller.parameters())
    type_clf_params = controller.type_clf.trainable_params
    gate_params = sum(p.numel() for p in controller.route_gate.parameters())
    print(f"Total params: {total:,}")
    print(f"  TypeClassifier: {type_clf_params:,}")
    print(f"  Route gate: {gate_params:,}")
    print(f"  (rest from factorization + reasoning)")

    # Forward pass (training mode)
    s0 = torch.randn(batch_size, hidden_dim)
    answer_emb = torch.randn(batch_size, n_answers, hidden_dim)
    labels = torch.randint(0, n_answers, (batch_size,))

    print("\n--- Training forward ---")
    result = controller(s0, answer_embeddings=answer_emb, labels=labels)
    print(f"  scores: {result['scores'].shape}")
    print(f"  detected_type: {result['detected_type'].tolist()}")
    print(f"  confidence: {[f'{c:.2f}' for c in result['confidence'].tolist()]}")
    print(f"  routing_weight: {[f'{w:.2f}' for w in result['routing_weight'].squeeze(-1).tolist()]}")
    print(f"  task_loss: {result['task_loss'].item():.4f}")
    print(f"  total_loss: {result['total_loss'].item():.4f}")

    # Gradient check
    result['total_loss'].backward()
    grad_count = sum(
        1 for p in controller.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_p = sum(1 for p in controller.parameters() if p.requires_grad)
    print(f"\nGradient flow: {grad_count}/{total_p} trainable params have gradients")

    # Inference forward
    print("\n--- Inference forward ---")
    controller.eval()
    with torch.no_grad():
        inf_result = controller.forward_inference(s0, answer_emb)
    print(f"  scores: {inf_result['scores'].shape}")
    print(f"  predicted: {inf_result['predicted'].tolist()}")
    print(f"  detected_type: {inf_result['detected_type'].tolist()}")
    print(f"  route_decisions: {inf_result['route_decisions'].tolist()}")
    print(f"    (0=fast, 1=structured)")

    # Diagnostics
    diag = controller.get_diagnostics(inf_result)
    print(f"\nDiagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Test merge_labels
    labels_10 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    labels_6 = merge_labels(labels_10)
    print(f"\nmerge_labels: {labels_10.tolist()} -> {labels_6.tolist()}")
    assert labels_6.tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]

    print("\nAll controller tests passed!")
