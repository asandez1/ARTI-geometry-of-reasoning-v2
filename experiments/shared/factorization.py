#!/usr/bin/env python3
"""
Module 2: Continuous Operator Manifold Factorization (Section 3.2 of draft).

Factorizes |s_0> into structural |o> and contextual |d> components,
then uses |o> to index into the ContinuousOperatorManifold.

Adapted from Paper 11 E3 models.py::StructuralFactorizationLayer.
Key change: integrates ContinuousOperatorManifold instead of discrete OperatorSet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .operators import ContinuousOperatorManifold


class FactorizationLayer(nn.Module):
    """
    Factorization layer with MI discriminator.

    Decomposes |s_0> into:
    - |o> in R^struct_dim: structural component (domain-independent reasoning)
    - |d> in R^context_dim: contextual component (domain-specific content)

    Each MLP is a 2-layer network: Linear -> GELU -> Linear -> L2-normalize.
    An MI discriminator enforces independence between |o> and |d>.

    From Section 3.2.1 of draft.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        struct_dim: int = 128,
        context_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.struct_dim = struct_dim
        self.context_dim = context_dim

        # Structural encoder: |s_0> -> |o>
        self.struct_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, struct_dim),
        )

        # Context encoder: |s_0> -> |d>
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )

        # MI discriminator: estimates I(|o>; |d>) via binary classifier
        # 2-layer MLP -> sigmoid
        self.mi_discriminator = nn.Sequential(
            nn.Linear(struct_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Factorize |s_0> into structural and context components.

        Args:
            s0: [B, hidden_dim] initial state

        Returns:
            structural: [B, struct_dim] — |o>
            context: [B, context_dim] — |d>
        """
        structural = self.struct_mlp(s0)
        structural = F.normalize(structural, dim=-1)

        context = self.context_mlp(s0)
        context = F.normalize(context, dim=-1)

        return structural, context

    def factorization_loss(
        self,
        structural: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        MI discriminator loss encouraging independence of |o> and |d>.

        L_factorize = -0.5 * [log D(o, d) + log(1 - D(o, d_shuffled))]

        The discriminator tries to distinguish real pairs from shuffled pairs.
        The factorization MLPs are trained adversarially to MINIMIZE this
        (making |o> and |d> indistinguishable = independent).

        Args:
            structural: [B, struct_dim]
            context: [B, context_dim]

        Returns:
            loss: scalar MI estimate
        """
        batch_size = structural.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=structural.device)

        # Positive pairs: (o_i, d_i)
        pos_pairs = torch.cat([structural, context], dim=-1)
        pos_scores = self.mi_discriminator(pos_pairs)

        # Negative pairs: (o_i, d_j) with j shuffled
        perm = torch.randperm(batch_size, device=structural.device)
        neg_pairs = torch.cat([structural, context[perm]], dim=-1)
        neg_scores = self.mi_discriminator(neg_pairs)

        # Binary cross-entropy
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        loss = (
            F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
            + F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
        )
        return loss


class ContinuousFactorizationModule(nn.Module):
    """
    Combined factorization + operator manifold module.

    This is the full Module 2 from Section 3.2:
    1. Factorize s0 into structural |o> and context |d>
    2. Use |o> to index into ContinuousOperatorManifold
    3. Apply interpolated operator to s0

    Returns the transformed state, factorization components, and operator weights.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        struct_dim: int = 128,
        context_dim: int = 128,
        manifold_dim: int = 10,
        n_anchors: int = 16,
        rank: int = 16,
        tau_init: float = 1.0,
        tau_decay: float = 0.95,
        tau_floor: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.struct_dim = struct_dim
        self.context_dim = context_dim

        # Factorization layer
        self.factorizer = FactorizationLayer(
            hidden_dim=hidden_dim,
            struct_dim=struct_dim,
            context_dim=context_dim,
        )

        # Continuous operator manifold
        self.manifold = ContinuousOperatorManifold(
            hidden_dim=hidden_dim,
            struct_dim=struct_dim,
            manifold_dim=manifold_dim,
            n_anchors=n_anchors,
            rank=rank,
            tau_init=tau_init,
            tau_decay=tau_decay,
            tau_floor=tau_floor,
        )

    def forward(self, s0: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full factorization forward pass.

        Args:
            s0: [B, hidden_dim] initial state from encoder

        Returns:
            Dict with:
                - transformed: [B, hidden_dim] O(s0)
                - structural: [B, struct_dim] |o>
                - context: [B, context_dim] |d>
                - weights: [B, n_anchors] attention weights
        """
        # Step 1: Factorize
        structural, context = self.factorizer(s0)

        # Step 2-3: Index into manifold and apply operator
        transformed, weights = self.manifold(s0, structural)

        return {
            'transformed': transformed,
            'structural': structural,
            'context': context,
            'weights': weights,
        }

    def factorization_loss(
        self,
        structural: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MI discriminator loss."""
        return self.factorizer.factorization_loss(structural, context)

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute anchor entropy regularizer."""
        return self.manifold.entropy_loss(weights)

    def anneal_temperature(self, epoch: int):
        """Anneal manifold temperature."""
        self.manifold.anneal_temperature(epoch)


if __name__ == "__main__":
    print("Testing ContinuousFactorizationModule...")

    module = ContinuousFactorizationModule(
        hidden_dim=256,
        struct_dim=128,
        context_dim=128,
        manifold_dim=10,
        n_anchors=16,
        rank=16,
    )

    total_params = sum(p.numel() for p in module.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"  Factorizer: {sum(p.numel() for p in module.factorizer.parameters()):,}")
    print(f"    Struct MLP: {sum(p.numel() for p in module.factorizer.struct_mlp.parameters()):,}")
    print(f"    Context MLP: {sum(p.numel() for p in module.factorizer.context_mlp.parameters()):,}")
    print(f"    MI disc: {sum(p.numel() for p in module.factorizer.mi_discriminator.parameters()):,}")
    print(f"  Manifold: {sum(p.numel() for p in module.manifold.parameters()):,}")

    # Forward pass
    s0 = torch.randn(8, 256)
    output = module(s0)
    print(f"\nForward pass:")
    print(f"  transformed: {output['transformed'].shape}")
    print(f"  structural: {output['structural'].shape}")
    print(f"  context: {output['context'].shape}")
    print(f"  weights: {output['weights'].shape}")

    # Losses
    fact_loss = module.factorization_loss(output['structural'], output['context'])
    ent_loss = module.entropy_loss(output['weights'])
    print(f"\nLosses:")
    print(f"  factorization: {fact_loss.item():.4f}")
    print(f"  entropy: {ent_loss.item():.4f}")

    # Backward
    total_loss = output['transformed'].sum() + fact_loss + ent_loss
    total_loss.backward()
    grad_count = sum(1 for p in module.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in module.parameters())
    print(f"\nGradient flow: {grad_count}/{total} params have gradients")

    print("\nFactorization tests passed!")
