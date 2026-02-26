#!/usr/bin/env python3
"""
Module 3: Reasoning Engine for CO-FRN (Section 3.3 of draft).

Adapted from Paper 11 E3 hilbert_tree.py::HilbertTreeFast.
Key change: uses ContinuousOperatorManifold instead of discrete OperatorSet.

For multi-step tasks (GSM8K, StrategyQA): HilbertTreeFast with beam search.
For single-step tasks (ARC, FOLIO): direct cosine scoring (no tree overhead).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class HilbertTreeFast(nn.Module):
    """
    Batched Hilbert tree for multi-step reasoning.

    Adapted from Paper 11 E3. Key differences:
    - Operator application via ContinuousOperatorManifold (soft attention)
      instead of discrete OperatorSet (Gumbel-softmax)
    - Default beam_width=2 (Paper 11 E3 ablations showed optimal)
    - Temperature annealing per depth: tau_tree = tau_0 * 0.9^depth

    The tree maintains a fixed beam of hypotheses, expands them via
    the operator manifold, scores against evidence, and soft-prunes.

    Coherence constraint P3:
        |<phi(v_i), phi(v_j)>| >= delta^(depth_diff)
    Enforced as a differentiable loss term.
    """

    def __init__(
        self,
        dim: int,
        beam_width: int = 2,
        max_depth: int = 3,
        delta: float = 0.8,
    ):
        super().__init__()
        self.dim = dim
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.delta = delta

        # Learned diversity vectors to break initial beam symmetry
        self.diversity_vectors = nn.Parameter(
            torch.randn(beam_width, dim) * 0.1
        )

    def forward(
        self,
        initial_state: torch.Tensor,
        evidence: torch.Tensor,
        operator_fn: callable,
        structural: torch.Tensor,
        depth: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Batched Hilbert tree reasoning pass.

        Args:
            initial_state: [B, dim] initial state (transformed s0)
            evidence: [B, dim] evidence vector for scoring
            operator_fn: callable(x, structural) -> (Ox, weights)
                The ContinuousOperatorManifold.forward method
            structural: [B, struct_dim] structural component for operator selection
            depth: reasoning depth (default: self.max_depth)

        Returns:
            Dict with:
                hypothesis_scores: [B, beam] final scores
                hypothesis_states: [B, beam, dim] final states
                selected_idx: [B] best hypothesis index
                coherence_loss: scalar P3 violation penalty
        """
        depth = depth or self.max_depth
        batch_size = initial_state.shape[0]
        device = initial_state.device

        # Initialize beam: replicate initial state with diversity
        # [B, beam, dim]
        states = initial_state.unsqueeze(1).expand(
            -1, self.beam_width, -1
        ).clone()
        diversity = self.diversity_vectors.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        states = F.normalize(states + diversity, dim=-1)

        weights = torch.ones(
            batch_size, self.beam_width, device=device
        ) / self.beam_width

        coherence_loss = torch.tensor(0.0, device=device)

        for d in range(depth):
            # Flatten beam for operator application
            flat_states = states.reshape(-1, self.dim)  # [B*beam, dim]

            # Expand structural for each beam element
            flat_struct = structural.unsqueeze(1).expand(
                -1, self.beam_width, -1
            ).reshape(-1, structural.shape[-1])  # [B*beam, struct_dim]

            # Apply continuous operator manifold
            new_flat, _ = operator_fn(flat_states, flat_struct)

            # Reshape back
            new_states = new_flat.reshape(
                batch_size, self.beam_width, self.dim
            )

            # Score against evidence
            evidence_exp = evidence.unsqueeze(1).expand_as(new_states)
            similarity = F.cosine_similarity(
                new_states, evidence_exp, dim=-1
            )  # [B, beam]
            scores = weights * similarity

            # Soft pruning via softmax re-weighting (differentiable)
            # Temperature anneals with depth
            prune_temp = max(0.05, 0.1 * (0.9 ** d))
            weights = F.softmax(scores / prune_temp, dim=-1)

            states = new_states

            # Coherence loss (sample subset for efficiency)
            n_sample = min(batch_size, 4)
            for b_idx in range(n_sample):
                s = states[b_idx]  # [beam, dim]
                inner = s @ s.T
                abs_inner = inner.abs()
                min_coh = self.delta ** (d + 1)
                mask = 1.0 - torch.eye(self.beam_width, device=device)
                violations = F.relu(min_coh - abs_inner) * mask
                coherence_loss = coherence_loss + violations.mean()

        if depth > 0:
            coherence_loss = coherence_loss / (depth * min(batch_size, 4))

        # Final scoring
        evidence_exp = evidence.unsqueeze(1).expand_as(states)
        final_sim = F.cosine_similarity(states, evidence_exp, dim=-1)
        final_scores = weights * final_sim

        return {
            'hypothesis_scores': final_scores,
            'hypothesis_states': states,
            'selected_idx': final_scores.argmax(dim=-1),
            'coherence_loss': coherence_loss,
        }


class DirectScorer(nn.Module):
    """
    Direct scoring head for single-step tasks (ARC, FOLIO).

    Two modes:
    - MLP mode (legacy): concat(transformed, answer) -> MLP -> scalar
    - Cosine mode (E10 fix): score = cos(transformed, answer) / tau

    The MLP mode was found to collapse to answer-only scoring (E9).
    Cosine mode cannot learn to ignore the transformed state.
    """

    def __init__(self, hidden_dim: int = 256, use_cosine: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cosine = use_cosine

        # MLP scorer (legacy)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable temperature for cosine scoring
        # Init at 0.1 so gradients through cosine are meaningful
        self.cosine_tau = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        transformed: torch.Tensor,
        answer_encodings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score answer candidates against the transformed state.

        Args:
            transformed: [B, hidden_dim] operator-transformed state O(s0)
            answer_encodings: [B, n_answers, hidden_dim] encoded answer choices

        Returns:
            scores: [B, n_answers] logits for each answer
        """
        if self.use_cosine:
            # Cosine similarity scoring — cannot ignore transformed
            t_exp = transformed.unsqueeze(1).expand_as(answer_encodings)
            cos_sim = F.cosine_similarity(t_exp, answer_encodings, dim=-1)
            # Scale by learnable temperature (clamped positive)
            tau = self.cosine_tau.abs().clamp(min=0.01)
            scores = cos_sim / tau
            return scores

        # Legacy MLP scoring
        t_exp = transformed.unsqueeze(1).expand_as(answer_encodings)
        paired = torch.cat([t_exp, answer_encodings], dim=-1)
        scores = self.scorer(paired).squeeze(-1)  # [B, n_answers]
        return scores


class ARTIRoutedScorer(nn.Module):
    """
    Scoring head that soft-blends cosine and direct classification
    via ARTI type probabilities as routing signal.

    Cosine mode works for content-rich answers (GSM8K math).
    Direct classification works for generic labels (Yes/No, True/False).
    The router learns which mode to use per-example based on ARTI's
    10-type probability vector.

    Architecture:
        ARTI(frozen) -> type_probs [B, 10]
        router: Linear(10, 1) + sigmoid -> alpha [B, 1]
        cosine_scores = cos(transformed, answer_enc) / tau
        direct_scores = MLP(transformed)
        final = alpha * cosine_scores + (1-alpha) * direct_scores
    """

    def __init__(self, hidden_dim: int = 256, max_choices: int = 4,
                 n_arti_types: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_choices = max_choices

        # Direct classification head (for generic labels)
        self.direct_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_choices),
        )

        # Learnable temperature for cosine scoring
        self.cosine_tau = nn.Parameter(torch.tensor(0.1))

        # Router: maps ARTI type probabilities to blend weight
        self.router = nn.Sequential(
            nn.Linear(n_arti_types, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        transformed: torch.Tensor,
        answer_encodings: torch.Tensor,
        arti_probs: torch.Tensor,
        valid_choices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score answer candidates with blended cosine + direct scoring.

        Args:
            transformed: [B, hidden_dim] operator-transformed state
            answer_encodings: [B, n_answers, hidden_dim] encoded answers
            arti_probs: [B, n_arti_types] ARTI type probabilities
            valid_choices: [B, n_answers] bool mask (True = valid)

        Returns:
            scores: [B, n_answers] blended logits
        """
        # Cosine scores
        t_exp = transformed.unsqueeze(1).expand_as(answer_encodings)
        cos_sim = F.cosine_similarity(t_exp, answer_encodings, dim=-1)
        tau = self.cosine_tau.abs().clamp(min=0.01)
        cos_scores = cos_sim / tau  # [B, n_answers]

        # Direct classification scores
        direct_scores = self.direct_head(transformed)  # [B, max_choices]
        # Slice to match n_answers if needed
        n_answers = answer_encodings.shape[1]
        direct_scores = direct_scores[:, :n_answers]

        # Route via ARTI type probabilities
        alpha = self.router(arti_probs)  # [B, 1]
        scores = alpha * cos_scores + (1 - alpha) * direct_scores

        # Mask invalid choices
        if valid_choices is not None:
            valid = valid_choices[:, :n_answers]
            scores = scores.masked_fill(~valid, -1e9)

        return scores, alpha


class ReasoningEngine(nn.Module):
    """
    Unified reasoning engine that selects between HilbertTree (multi-step)
    and DirectScorer (single-step) based on task type.

    This wraps both reasoning modes and exposes a common interface.

    E10 fix: use_cosine_scoring=True replaces MLP-based tree_scorer and
    DirectScorer with cosine similarity scoring, preventing the collapse
    to answer-only scoring discovered in E9.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        beam_width: int = 2,
        max_depth: int = 3,
        delta: float = 0.8,
        use_cosine_scoring: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_cosine_scoring = use_cosine_scoring

        # Multi-step reasoning
        self.tree = HilbertTreeFast(
            dim=hidden_dim,
            beam_width=beam_width,
            max_depth=max_depth,
            delta=delta,
        )

        # Single-step scoring
        self.direct = DirectScorer(
            hidden_dim=hidden_dim,
            use_cosine=use_cosine_scoring,
        )

        # Answer projection for tree mode: best hypothesis -> answer score
        # Legacy MLP scorer (kept for backward compat with saved models)
        self.tree_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable temperature for cosine tree scoring (E10)
        self.tree_cosine_tau = nn.Parameter(torch.tensor(0.1))

    def forward_multistep(
        self,
        transformed: torch.Tensor,
        evidence: torch.Tensor,
        operator_fn: callable,
        structural: torch.Tensor,
        answer_encodings: torch.Tensor,
        depth: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-step reasoning via Hilbert tree.

        Args:
            transformed: [B, hidden_dim] O(s0)
            evidence: [B, hidden_dim] evidence for tree scoring
            operator_fn: manifold operator callable
            structural: [B, struct_dim] for operator selection
            answer_encodings: [B, n_answers, hidden_dim]
            depth: reasoning depth

        Returns:
            Dict with scores, coherence_loss, tree outputs
        """
        tree_out = self.tree(
            initial_state=transformed,
            evidence=evidence,
            operator_fn=operator_fn,
            structural=structural,
            depth=depth,
        )

        # Get best hypothesis as weighted sum of beam
        h_weights = F.softmax(
            tree_out['hypothesis_scores'], dim=-1
        )  # [B, beam]
        best_hyp = (
            h_weights.unsqueeze(-1) * tree_out['hypothesis_states']
        ).sum(dim=1)  # [B, hidden_dim]

        # Score answers against best hypothesis
        if self.use_cosine_scoring:
            # E10 fix: cosine scoring — cannot ignore best_hyp
            best_exp = best_hyp.unsqueeze(1).expand_as(answer_encodings)
            cos_sim = F.cosine_similarity(best_exp, answer_encodings, dim=-1)
            tau = self.tree_cosine_tau.abs().clamp(min=0.01)
            scores = cos_sim / tau
        else:
            # Legacy MLP scoring (vulnerable to collapse — see E9)
            best_exp = best_hyp.unsqueeze(1).expand_as(answer_encodings)
            paired = torch.cat([best_exp, answer_encodings], dim=-1)
            scores = self.tree_scorer(paired).squeeze(-1)

        return {
            'scores': scores,
            'coherence_loss': tree_out['coherence_loss'],
            'hypothesis_states': tree_out['hypothesis_states'],
            'hypothesis_scores': tree_out['hypothesis_scores'],
        }

    def forward_direct(
        self,
        transformed: torch.Tensor,
        answer_encodings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-step direct scoring.

        Args:
            transformed: [B, hidden_dim] O(s0)
            answer_encodings: [B, n_answers, hidden_dim]

        Returns:
            Dict with scores and zero coherence_loss
        """
        scores = self.direct(transformed, answer_encodings)
        return {
            'scores': scores,
            'coherence_loss': torch.tensor(
                0.0, device=transformed.device
            ),
        }


if __name__ == "__main__":
    print("Testing ReasoningEngine...")

    hidden_dim = 256
    struct_dim = 128
    batch_size = 4
    n_answers = 4

    engine = ReasoningEngine(
        hidden_dim=hidden_dim,
        beam_width=2,
        max_depth=3,
    )

    total_params = sum(p.numel() for p in engine.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test direct scoring
    transformed = F.normalize(torch.randn(batch_size, hidden_dim), dim=-1)
    answers = F.normalize(torch.randn(batch_size, n_answers, hidden_dim), dim=-1)

    direct_out = engine.forward_direct(transformed, answers)
    print(f"\nDirect scoring:")
    print(f"  scores: {direct_out['scores'].shape}")
    print(f"  coherence_loss: {direct_out['coherence_loss'].item():.4f}")

    # Test multistep (mock operator_fn)
    def mock_operator_fn(x, s):
        return F.normalize(x + 0.1 * torch.randn_like(x), dim=-1), None

    evidence = F.normalize(torch.randn(batch_size, hidden_dim), dim=-1)
    structural = F.normalize(torch.randn(batch_size, struct_dim), dim=-1)

    tree_out = engine.forward_multistep(
        transformed, evidence, mock_operator_fn, structural, answers, depth=2
    )
    print(f"\nMultistep (tree) scoring:")
    print(f"  scores: {tree_out['scores'].shape}")
    print(f"  coherence_loss: {tree_out['coherence_loss'].item():.4f}")
    print(f"  hypothesis_states: {tree_out['hypothesis_states'].shape}")

    # Gradient check
    loss = direct_out['scores'].sum()
    loss.backward()
    grad_count = sum(
        1 for p in engine.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for p in engine.parameters())
    print(f"\nGradient flow (direct): {grad_count}/{total}")

    print("\nReasoningEngine tests passed!")
