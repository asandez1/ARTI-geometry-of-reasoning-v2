#!/usr/bin/env python3
"""
ARTI v2: Trajectory-Based Reasoning Type Classification.

Classifies reasoning types from trajectory geometry on the 10D operator
manifold, rather than single-point position (v1).

Architecture:
    Text -> segment_text() -> [clause_1, ..., clause_n]
         -> SentenceTransformerEncoder -> [emb_1, ..., emb_n]  (384D each, frozen)
         -> ManifoldProjection -> [coord_1, ..., coord_n]       (10D each, frozen from v1)
         -> TrajectoryFeatureExtractor -> 60D raw -> MLP -> 48D
         -> Classifier(48 -> 96 -> 10)

Key insight (Paper 13): Reasoning types are distinguished by trajectory
geometry -- how the path moves through the manifold, not where a single
embedding lands. PhysCause = linear, Induction = converging, Analogy = parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .reasoning_types import ReasoningType, N_REASONING_TYPES, REASONING_TYPES
from .arti import ManifoldProjection


@dataclass
class ARTIV2Config:
    """Configuration for ARTI v2 (trajectory-based)."""
    encoder_dim: int = 384          # Sentence-transformer output dim
    manifold_dim: int = 10          # Manifold projection dim (frozen from v1)
    traj_feature_dim: int = 60      # Raw trajectory feature vector size
    traj_hidden: int = 48           # Trajectory MLP output dim
    n_classes: int = N_REASONING_TYPES
    classifier_hidden: int = 96     # Classifier hidden dim
    dropout: float = 0.1
    # How many step deltas and curvatures to include (zero-padded)
    max_step_deltas: int = 3        # First 3 step deltas (10D each = 30D)
    max_curvatures: int = 9         # First 9 curvatures


class TrajectoryFeatureExtractor(nn.Module):
    """
    Extracts fixed-size trajectory features from variable-length manifold
    coordinate sequences.

    Input: list of [n_steps, manifold_dim] coordinate tensors (variable n_steps)
    Output: [B, traj_hidden] trajectory feature vectors

    Raw feature groups (60D total):
        - Global shape: total_displacement, path_length, displacement_ratio, n_steps  (4D)
        - Curvature stats: mean/max/std curvature, curvature_range                   (4D)
        - Direction consistency: mean/min cosine, direction_changes                    (3D)
        - Shape descriptors: linearity, loop_score, convergence, parallelism,
                            jump_score, cascade_score                                 (6D)
        - Per-step stats: mean/std delta magnitude, mean/std coord values             (4D)
        - Padded step deltas: first 3 step deltas (10D each), zero-padded            (30D)
        - Padded curvatures: first 9 curvatures, zero-padded                          (9D)
                                                                            Total:    60D
    """

    def __init__(self, config: ARTIV2Config):
        super().__init__()
        self.config = config
        self.manifold_dim = config.manifold_dim
        self.max_step_deltas = config.max_step_deltas
        self.max_curvatures = config.max_curvatures

        # MLP: raw features -> compact representation
        self.mlp = nn.Sequential(
            nn.Linear(config.traj_feature_dim, config.traj_hidden),
            nn.LayerNorm(config.traj_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def _extract_raw_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Extract 60D raw trajectory features from a single sequence of
        manifold coordinates.

        Args:
            coords: [n_steps, manifold_dim] manifold coordinates

        Returns:
            features: [60] raw feature vector
        """
        device = coords.device
        n_steps = coords.shape[0]
        md = self.manifold_dim

        features = []

        # === Deltas ===
        if n_steps >= 2:
            deltas = coords[1:] - coords[:-1]  # [n_steps-1, md]
            delta_mags = deltas.norm(dim=-1)    # [n_steps-1]
        else:
            deltas = torch.zeros(1, md, device=device)
            delta_mags = torch.zeros(1, device=device)

        # === Global shape (4D) ===
        total_displacement = (coords[-1] - coords[0]).norm()
        path_length = delta_mags.sum()
        displacement_ratio = total_displacement / (path_length + 1e-9)
        n_steps_norm = torch.tensor(n_steps / 10.0, device=device)  # Normalize
        features.extend([total_displacement, path_length, displacement_ratio, n_steps_norm])

        # === Curvature stats (4D) ===
        if n_steps >= 3:
            curvatures = []
            for i in range(len(deltas) - 1):
                cos_sim = F.cosine_similarity(
                    deltas[i].unsqueeze(0), deltas[i + 1].unsqueeze(0)
                ).squeeze()
                curvature = 1.0 - cos_sim  # 0 = straight, 2 = reversal
                curvatures.append(curvature)
            curvatures_t = torch.stack(curvatures)
            mean_curv = curvatures_t.mean()
            max_curv = curvatures_t.max()
            std_curv = curvatures_t.std() if len(curvatures) > 1 else torch.tensor(0.0, device=device)
            curv_range = max_curv - curvatures_t.min()
        else:
            curvatures_t = torch.zeros(1, device=device)
            mean_curv = torch.tensor(0.0, device=device)
            max_curv = torch.tensor(0.0, device=device)
            std_curv = torch.tensor(0.0, device=device)
            curv_range = torch.tensor(0.0, device=device)
        features.extend([mean_curv, max_curv, std_curv, curv_range])

        # === Direction consistency (3D) ===
        if n_steps >= 3:
            cosines = []
            for i in range(len(deltas) - 1):
                c = F.cosine_similarity(
                    deltas[i].unsqueeze(0), deltas[i + 1].unsqueeze(0)
                ).squeeze()
                cosines.append(c)
            cosines_t = torch.stack(cosines)
            mean_cos = cosines_t.mean()
            min_cos = cosines_t.min()
            # Count direction changes (cosine < 0)
            dir_changes = (cosines_t < 0).float().sum() / max(len(cosines), 1)
        else:
            mean_cos = torch.tensor(1.0, device=device)
            min_cos = torch.tensor(1.0, device=device)
            dir_changes = torch.tensor(0.0, device=device)
        features.extend([mean_cos, min_cos, dir_changes])

        # === Shape descriptors (6D) ===
        # Linearity: displacement / path_length (1.0 = straight line)
        linearity = displacement_ratio

        # Loop score: how close end is to start (relative to path length)
        loop_score = 1.0 - total_displacement / (path_length + 1e-9)

        # Convergence score: do deltas get smaller? (positive = converging)
        if n_steps >= 3:
            first_half = delta_mags[:len(delta_mags) // 2 + 1].mean()
            second_half = delta_mags[len(delta_mags) // 2:].mean()
            convergence = (first_half - second_half) / (first_half + 1e-9)
        else:
            convergence = torch.tensor(0.0, device=device)

        # Parallelism score: mean pairwise cosine of all delta pairs
        if n_steps >= 3 and len(deltas) >= 2:
            all_cos = []
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    c = F.cosine_similarity(
                        deltas[i].unsqueeze(0), deltas[j].unsqueeze(0)
                    ).squeeze()
                    all_cos.append(c)
            parallelism = torch.stack(all_cos).mean()
        else:
            parallelism = torch.tensor(0.0, device=device)

        # Jump score: max delta magnitude / mean delta magnitude
        if delta_mags.numel() > 0 and delta_mags.mean() > 1e-9:
            jump_score = delta_mags.max() / delta_mags.mean()
        else:
            jump_score = torch.tensor(1.0, device=device)

        # Cascade score: are deltas growing? (positive = cascading/amplifying)
        if n_steps >= 3:
            cascade = (second_half - first_half) / (first_half + 1e-9)
        else:
            cascade = torch.tensor(0.0, device=device)

        features.extend([linearity, loop_score, convergence, parallelism, jump_score, cascade])

        # === Per-step stats (4D) ===
        mean_delta_mag = delta_mags.mean()
        std_delta_mag = delta_mags.std() if delta_mags.numel() > 1 else torch.tensor(0.0, device=device)
        mean_coord = coords.mean()
        std_coord = coords.std()
        features.extend([mean_delta_mag, std_delta_mag, mean_coord, std_coord])

        # === Padded step deltas (30D) ===
        padded_deltas = torch.zeros(self.max_step_deltas * md, device=device)
        n_deltas_to_copy = min(len(deltas), self.max_step_deltas)
        if n_deltas_to_copy > 0:
            padded_deltas[:n_deltas_to_copy * md] = deltas[:n_deltas_to_copy].reshape(-1)
        features.append(padded_deltas)

        # === Padded curvatures (9D) ===
        padded_curvs = torch.zeros(self.max_curvatures, device=device)
        if n_steps >= 3:
            n_curvs_to_copy = min(len(curvatures_t), self.max_curvatures)
            padded_curvs[:n_curvs_to_copy] = curvatures_t[:n_curvs_to_copy]
        features.append(padded_curvs)

        # Concatenate all scalar features + padded vectors
        scalar_features = torch.stack([f if isinstance(f, torch.Tensor) else torch.tensor(f, device=device)
                                        for f in features[:-2]])  # All except the last 2 padded vectors
        return torch.cat([scalar_features, features[-2], features[-1]])  # [60]

    def forward(self, coords_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract trajectory features for a batch.

        Args:
            coords_list: list of [n_steps_i, manifold_dim] tensors
                         (variable n_steps per sample)

        Returns:
            features: [B, traj_hidden] trajectory feature vectors
        """
        raw_features = torch.stack([
            self._extract_raw_features(coords) for coords in coords_list
        ])  # [B, 60]
        return self.mlp(raw_features)  # [B, traj_hidden]


class ARTIV2(nn.Module):
    """
    ARTI v2: Trajectory-Based Reasoning Type Classification.

    Pipeline:
        clause_embeddings -> ManifoldProjection (frozen) -> coords
                          -> TrajectoryFeatureExtractor -> traj_features
                          -> Classifier -> logits

    The ManifoldProjection is loaded from v1 and frozen. Only the
    TrajectoryFeatureExtractor MLP and Classifier are trained.
    """

    def __init__(self, config: ARTIV2Config):
        super().__init__()
        self.config = config

        # Manifold projection (loaded from v1, frozen)
        self.manifold_proj = ManifoldProjection(
            input_dim=config.encoder_dim,
            manifold_dim=config.manifold_dim,
        )

        # Trajectory feature extraction
        self.traj_extractor = TrajectoryFeatureExtractor(config)

        # Classifier: traj_hidden -> classifier_hidden -> n_classes
        self.classifier = nn.Sequential(
            nn.Linear(config.traj_hidden, config.classifier_hidden),
            nn.LayerNorm(config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, config.n_classes),
        )

    def freeze_manifold(self):
        """Freeze the manifold projection (transfer from v1)."""
        for param in self.manifold_proj.parameters():
            param.requires_grad = False

    def forward(
        self,
        clause_embeddings_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Classify reasoning type from clause embeddings.

        Args:
            clause_embeddings_list: list of [n_clauses_i, encoder_dim] tensors
                                   (variable n_clauses per sample)

        Returns:
            Dict with logits, probabilities, type, confidence, traj_features
        """
        # Project each sample's clause embeddings through manifold
        coords_list = []
        for clause_embs in clause_embeddings_list:
            with torch.no_grad():
                coords = self.manifold_proj(clause_embs)  # [n_clauses, manifold_dim]
            coords_list.append(coords)

        # Extract trajectory features
        traj_features = self.traj_extractor(coords_list)  # [B, traj_hidden]

        # Classify
        logits = self.classifier(traj_features)  # [B, n_classes]
        probs = F.softmax(logits, dim=-1)

        return {
            'logits': logits,
            'probabilities': probs,
            'type': probs.argmax(dim=-1),
            'confidence': probs.max(dim=-1).values,
            'traj_features': traj_features,
        }

    def identify(self, clause_embeddings_list: List[torch.Tensor]) -> Dict[str, object]:
        """Convenience method: identify reasoning types with human-readable output."""
        with torch.no_grad():
            result = self.forward(clause_embeddings_list)

        type_indices = result['type'].cpu().numpy()
        type_names = [ReasoningType(int(i)).name for i in type_indices]
        probs = result['probabilities'].cpu().numpy()

        breakdowns = []
        for b in range(len(type_indices)):
            bd = {
                REASONING_TYPES[ReasoningType(i)].short_name: float(probs[b, i])
                for i in range(N_REASONING_TYPES)
            }
            breakdowns.append(bd)

        return {
            'type_names': type_names,
            'type_indices': type_indices.tolist(),
            'confidence': result['confidence'].cpu().numpy().tolist(),
            'breakdowns': breakdowns,
        }

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_breakdown(self) -> Dict[str, int]:
        def count(module, trainable_only=True):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        return {
            'manifold_projection (frozen)': count(self.manifold_proj, trainable_only=False),
            'manifold_projection (trainable)': count(self.manifold_proj),
            'trajectory_extractor': count(self.traj_extractor),
            'classifier': count(self.classifier),
            'total_trainable': self.trainable_params,
            'total_all': sum(p.numel() for p in self.parameters()),
        }


if __name__ == "__main__":
    print("Testing ARTI v2 module...")

    config = ARTIV2Config()
    model = ARTIV2(config)

    breakdown = model.get_param_breakdown()
    print(f"\nParameter breakdown:")
    for k, v in breakdown.items():
        print(f"  {k}: {v:,}")

    # Freeze manifold and check
    model.freeze_manifold()
    print(f"\nAfter freezing manifold:")
    print(f"  Trainable: {model.trainable_params:,}")

    # Forward pass with variable-length clause sequences
    batch = [
        torch.randn(3, 384),   # 3 clauses
        torch.randn(5, 384),   # 5 clauses
        torch.randn(2, 384),   # 2 clauses
        torch.randn(7, 384),   # 7 clauses
    ]
    result = model(batch)
    print(f"\nForward pass (batch of 4):")
    print(f"  logits: {result['logits'].shape}")
    print(f"  types: {result['type'].tolist()}")
    print(f"  confidence: {[f'{c:.2f}' for c in result['confidence'].tolist()]}")
    print(f"  traj_features: {result['traj_features'].shape}")

    # Test gradient flow
    result['logits'].sum().backward()
    grad_count = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\nGradient flow: {grad_count}/{total} trainable params have gradients")

    # Test identify
    readable = model.identify(batch)
    print(f"\nIdentified types: {readable['type_names']}")

    print("\nAll ARTI v2 tests passed!")
