#!/usr/bin/env python3
"""
Active Reasoning Type Identifier (ARTI).

A lightweight module (<50K parameters) that identifies the type of reasoning
in a text passage by its geometric signature on the operator manifold.

Architecture:
    text -> FrozenEncoder -> embed [B, D]
         -> compute deltas (rolling window) -> Δ [B, D]
         -> ManifoldProjection -> manifold_coord [B, 10]
         -> GeometricFeatures -> features [B, F]
         -> TypeClassifier -> probabilities [B, 8]

The module outputs a probability distribution over 8 reasoning types:
    Cause-Effect, Deduction, Induction, Analogy, Conservation,
    Counterfactual, Abduction, Decomposition

Three operating modes:
    1. 'centroid' — Zero-shot: nearest centroid in manifold space (no training)
    2. 'linear'  — Minimal: logistic regression on manifold coords (tiny training)
    3. 'mlp'     — Full: 2-layer MLP on geometric features (~25K params)

Integrates with existing CO-FRN infrastructure:
    - Uses SentenceTransformerEncoder / PrecomputedEncoder for embeddings
    - Can optionally reuse ContinuousOperatorManifold's anchors + projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .reasoning_types import ReasoningType, N_REASONING_TYPES, REASONING_TYPES


@dataclass
class ARTIConfig:
    """Configuration for the Active Reasoning Type Identifier."""
    # Input dimensions
    encoder_dim: int = 384          # Sentence-transformer output dim
    manifold_dim: int = 10          # Projection to manifold space

    # Geometric feature extraction
    n_geometric_features: int = 32  # Total geometric feature vector size

    # Classifier
    classifier_type: str = 'mlp'    # 'centroid', 'linear', 'mlp'
    hidden_dim: int = 64            # MLP hidden dim (only for 'mlp')
    dropout: float = 0.1

    # Delta computation
    window_size: int = 2            # Rolling window for delta vectors

    # Temperature for output softmax
    temperature: float = 1.0


class ManifoldProjection(nn.Module):
    """
    Projects encoder embeddings onto the ~10D reasoning manifold.

    Can either:
    - Learn its own projection (fresh training)
    - Reuse CO-FRN's manifold_proj + anchors (transfer from Paper 12)
    """

    def __init__(self, input_dim: int, manifold_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.manifold_dim = manifold_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, manifold_dim * 2),
            nn.GELU(),
            nn.Linear(manifold_dim * 2, manifold_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to manifold coordinates.

        Args:
            x: [B, input_dim] encoder embeddings or delta vectors

        Returns:
            coords: [B, manifold_dim] manifold coordinates
        """
        return self.projection(x)

    @classmethod
    def from_cofrn(cls, cofrn_model) -> 'ManifoldProjection':
        """
        Initialize from a trained CO-FRN model's manifold projection.

        Extracts the struct_mlp -> manifold_proj chain from CO-FRN's
        ContinuousFactorizationModule.
        """
        manifold = cofrn_model.factorization.manifold
        struct_mlp = cofrn_model.factorization.factorizer.struct_mlp

        # Build combined projection: hidden_dim -> struct_dim -> manifold_dim
        hidden_dim = struct_mlp[0].in_features
        manifold_dim = manifold.manifold_dim

        proj = cls(input_dim=hidden_dim, manifold_dim=manifold_dim)

        # Copy weights from struct_mlp -> manifold_proj chain
        # (This is approximate — the full chain is struct_mlp + normalize + manifold_proj)
        with torch.no_grad():
            proj.projection[0].weight.copy_(
                manifold.manifold_proj.weight @ struct_mlp[2].weight
            )

        return proj


class GeometricFeatureExtractor(nn.Module):
    """
    Extracts geometric features from manifold coordinates and delta vectors.

    Computes features that capture the geometric signature of each reasoning type:
    - Manifold coordinates (10D)
    - Delta magnitude
    - Delta direction (unit vector, 10D)
    - Curvature estimate (angle between consecutive deltas)
    - Direction consistency (cosine similarity with running mean)
    """

    def __init__(self, manifold_dim: int = 10, n_features: int = 32):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.n_features = n_features

        # Learned feature mixing (combines raw geometric features into n_features)
        raw_feature_dim = manifold_dim + 1 + manifold_dim + 3  # coords + mag + direction + stats
        self.feature_mlp = nn.Sequential(
            nn.Linear(raw_feature_dim, n_features),
            nn.GELU(),
        )

    def forward(
        self,
        manifold_coords: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
        prev_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract geometric features.

        Args:
            manifold_coords: [B, manifold_dim] current position on manifold
            delta: [B, manifold_dim] current delta vector (optional)
            prev_delta: [B, manifold_dim] previous delta vector (optional, for curvature)

        Returns:
            features: [B, n_features] geometric feature vector
        """
        B = manifold_coords.shape[0]
        device = manifold_coords.device

        parts = [manifold_coords]  # [B, manifold_dim]

        # Delta magnitude
        if delta is not None:
            mag = delta.norm(dim=-1, keepdim=True)  # [B, 1]
            parts.append(mag)

            # Delta direction (unit vector)
            direction = F.normalize(delta, dim=-1)  # [B, manifold_dim]
            parts.append(direction)
        else:
            parts.append(torch.zeros(B, 1, device=device))
            parts.append(torch.zeros(B, self.manifold_dim, device=device))

        # Curvature: angle between consecutive deltas
        if delta is not None and prev_delta is not None:
            cos_angle = F.cosine_similarity(delta, prev_delta, dim=-1)  # [B]
            curvature = 1.0 - cos_angle  # 0 = straight, 2 = reversal
            parts.append(curvature.unsqueeze(-1))
        else:
            parts.append(torch.zeros(B, 1, device=device))

        # Displacement from origin
        dist_from_origin = manifold_coords.norm(dim=-1, keepdim=True)
        parts.append(dist_from_origin)

        # Variance of coordinates (spread indicator)
        coord_var = manifold_coords.var(dim=-1, keepdim=True)
        parts.append(coord_var)

        raw_features = torch.cat(parts, dim=-1)  # [B, raw_feature_dim]
        return self.feature_mlp(raw_features)


class ReasoningTypeClassifier(nn.Module):
    """
    Classifies reasoning type from geometric features.

    Three modes:
    - 'centroid': No parameters — assigns to nearest type centroid
    - 'linear': Single linear layer (10D -> 8 types)
    - 'mlp': 2-layer MLP with dropout
    """

    def __init__(
        self,
        input_dim: int = 32,
        n_types: int = N_REASONING_TYPES,
        classifier_type: str = 'mlp',
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier_type = classifier_type
        self.n_types = n_types

        if classifier_type == 'linear':
            self.classifier = nn.Linear(input_dim, n_types)
        elif classifier_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_types),
            )
        elif classifier_type == 'centroid':
            # Learnable centroids (initialized, then can be frozen)
            self.centroids = nn.Parameter(
                torch.randn(n_types, input_dim) * 0.1
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify reasoning type.

        Args:
            features: [B, input_dim] geometric features

        Returns:
            logits: [B, n_types] raw scores (apply softmax externally)
        """
        if self.classifier_type == 'centroid':
            # Negative distance to each centroid
            dists = torch.cdist(features.unsqueeze(0), self.centroids.unsqueeze(0))
            return -dists.squeeze(0)  # [B, n_types]
        else:
            return self.classifier(features)


class ARTI(nn.Module):
    """
    Active Reasoning Type Identifier.

    Full pipeline:
        text/embedding -> manifold projection -> geometric features -> type classification

    Usage:
        arti = ARTI(config)
        result = arti(embeddings)
        # result['type'] = argmax type index [B]
        # result['probabilities'] = soft distribution [B, 8]
        # result['confidence'] = max probability [B]
        # result['manifold_coords'] = 10D coordinates [B, 10]
        # result['features'] = geometric features [B, 32]
    """

    def __init__(self, config: ARTIConfig):
        super().__init__()
        self.config = config

        # Manifold projection
        self.manifold_proj = ManifoldProjection(
            input_dim=config.encoder_dim,
            manifold_dim=config.manifold_dim,
        )

        # Geometric feature extraction
        self.feature_extractor = GeometricFeatureExtractor(
            manifold_dim=config.manifold_dim,
            n_features=config.n_geometric_features,
        )

        # Type classifier
        self.type_classifier = ReasoningTypeClassifier(
            input_dim=config.n_geometric_features,
            n_types=N_REASONING_TYPES,
            classifier_type=config.classifier_type,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        prev_embeddings: Optional[torch.Tensor] = None,
        prev_prev_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Identify reasoning type from embeddings.

        For single-step classification (most common use):
            Pass only `embeddings` — classifies each embedding independently.

        For trajectory-aware classification (delta-based):
            Pass `embeddings` + `prev_embeddings` to compute delta vectors.
            Optionally pass `prev_prev_embeddings` for curvature estimation.

        Args:
            embeddings: [B, encoder_dim] current step embeddings
            prev_embeddings: [B, encoder_dim] previous step (optional)
            prev_prev_embeddings: [B, encoder_dim] two steps back (optional)

        Returns:
            Dict with type, probabilities, confidence, manifold_coords, features
        """
        # Project to manifold
        manifold_coords = self.manifold_proj(embeddings)

        # Compute deltas if trajectory info available
        delta = None
        prev_delta = None
        if prev_embeddings is not None:
            delta_raw = embeddings - prev_embeddings
            delta = self.manifold_proj(delta_raw)
        if prev_embeddings is not None and prev_prev_embeddings is not None:
            prev_delta_raw = prev_embeddings - prev_prev_embeddings
            prev_delta = self.manifold_proj(prev_delta_raw)

        # Extract geometric features
        features = self.feature_extractor(
            manifold_coords, delta=delta, prev_delta=prev_delta
        )

        # Classify
        logits = self.type_classifier(features)
        probs = F.softmax(logits / self.config.temperature, dim=-1)

        return {
            'type': probs.argmax(dim=-1),
            'probabilities': probs,
            'confidence': probs.max(dim=-1).values,
            'logits': logits,
            'manifold_coords': manifold_coords,
            'features': features,
        }

    def identify(self, embeddings: torch.Tensor) -> Dict[str, object]:
        """
        Convenience method: identify reasoning types with human-readable output.

        Args:
            embeddings: [B, encoder_dim]

        Returns:
            Dict with type_names, probabilities, confidence, coords
        """
        with torch.no_grad():
            result = self.forward(embeddings)

        type_indices = result['type'].cpu().numpy()
        type_names = [ReasoningType(int(i)).name for i in type_indices]
        probs = result['probabilities'].cpu().numpy()

        # Build per-sample breakdown
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
            'manifold_coords': result['manifold_coords'].cpu().numpy(),
        }

    @classmethod
    def from_pretrained_manifold(
        cls,
        cofrn_model,
        config: Optional[ARTIConfig] = None,
    ) -> 'ARTI':
        """
        Initialize ARTI using a trained CO-FRN model's manifold.

        Transfers the learned manifold projection so the ARTI classifier
        operates in the same geometric space as the trained reasoning model.
        """
        if config is None:
            config = ARTIConfig(
                encoder_dim=cofrn_model.config.hidden_dim,
                manifold_dim=cofrn_model.config.manifold_dim,
            )

        arti = cls(config)

        # Transfer manifold projection weights
        arti.manifold_proj = ManifoldProjection.from_cofrn(cofrn_model)

        return arti

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_breakdown(self) -> Dict[str, int]:
        return {
            'manifold_projection': sum(
                p.numel() for p in self.manifold_proj.parameters()
            ),
            'feature_extractor': sum(
                p.numel() for p in self.feature_extractor.parameters()
            ),
            'type_classifier': sum(
                p.numel() for p in self.type_classifier.parameters()
            ),
            'total': self.trainable_params,
        }


# ─── Streaming Identifier ─────────────────────────────────────────────────────

class StreamingARTI:
    """
    Stateful wrapper for real-time reasoning type tracking.

    Maintains a rolling window of embeddings for delta computation
    and returns type identification at each step.

    Usage:
        stream = StreamingARTI(arti_model, encoder)
        for text_step in reasoning_trace:
            result = stream.step(text_step)
            print(f"Step: {result['type_name']} ({result['confidence']:.0%})")
        history = stream.get_history()
    """

    def __init__(
        self,
        arti: ARTI,
        encoder=None,
        window_size: int = 3,
    ):
        self.arti = arti
        self.encoder = encoder
        self.window_size = window_size

        # Rolling buffer of recent embeddings
        self._buffer: List[torch.Tensor] = []
        self._history: List[Dict] = []

    def reset(self):
        """Clear the rolling buffer and history."""
        self._buffer = []
        self._history = []

    @torch.no_grad()
    def step(
        self,
        text: Optional[str] = None,
        embedding: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Process one reasoning step.

        Args:
            text: reasoning step text (requires encoder)
            embedding: pre-computed embedding [encoder_dim]

        Returns:
            Dict with type_name, confidence, breakdown, coords
        """
        if embedding is None:
            if self.encoder is None or text is None:
                raise ValueError("Provide either embedding or (text + encoder)")
            embedding = self._encode_text(text)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # Update buffer
        self._buffer.append(embedding)
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size:]

        # Build inputs
        prev = self._buffer[-2] if len(self._buffer) >= 2 else None
        prev_prev = self._buffer[-3] if len(self._buffer) >= 3 else None

        # Identify
        result = self.arti(
            embeddings=embedding,
            prev_embeddings=prev,
            prev_prev_embeddings=prev_prev,
        )

        type_idx = result['type'].item()
        rtype = ReasoningType(type_idx)

        step_result = {
            'step': len(self._history),
            'type': rtype,
            'type_name': REASONING_TYPES[rtype].name,
            'short_name': REASONING_TYPES[rtype].short_name,
            'confidence': result['confidence'].item(),
            'probabilities': result['probabilities'].squeeze(0).cpu().numpy(),
            'manifold_coords': result['manifold_coords'].squeeze(0).cpu().numpy(),
        }

        self._history.append(step_result)
        return step_result

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text through the encoder."""
        if hasattr(self.encoder, 'encode_texts'):
            return self.encoder.encode_texts([text]).squeeze(0)
        elif hasattr(self.encoder, 'encode'):
            return self.encoder.encode([text], convert_to_tensor=True).squeeze(0)
        else:
            raise ValueError(f"Unknown encoder type: {type(self.encoder)}")

    def get_history(self) -> List[Dict]:
        """Get full history of type identifications."""
        return self._history

    def get_trajectory(self) -> np.ndarray:
        """Get manifold coordinate trajectory [n_steps, manifold_dim]."""
        if not self._history:
            return np.array([])
        return np.stack([h['manifold_coords'] for h in self._history])

    def get_type_sequence(self) -> List[str]:
        """Get sequence of identified type names."""
        return [h['short_name'] for h in self._history]


if __name__ == "__main__":
    print("Testing ARTI module...")

    config = ARTIConfig(
        encoder_dim=384,
        manifold_dim=10,
        n_geometric_features=32,
        classifier_type='mlp',
        hidden_dim=64,
    )

    arti = ARTI(config)
    breakdown = arti.get_param_breakdown()
    print(f"\nParameter breakdown:")
    for k, v in breakdown.items():
        print(f"  {k}: {v:,}")

    # Forward pass
    batch_size = 8
    embeddings = torch.randn(batch_size, 384)
    result = arti(embeddings)
    print(f"\nForward pass:")
    print(f"  type: {result['type'].tolist()}")
    print(f"  confidence: {result['confidence'].tolist()}")
    print(f"  probabilities shape: {result['probabilities'].shape}")
    print(f"  manifold_coords shape: {result['manifold_coords'].shape}")

    # Human-readable output
    readable = arti.identify(embeddings)
    print(f"\nIdentified types: {readable['type_names'][:3]}...")

    # Test all classifier types
    for clf_type in ['centroid', 'linear', 'mlp']:
        cfg = ARTIConfig(classifier_type=clf_type)
        m = ARTI(cfg)
        out = m(embeddings)
        print(f"  {clf_type}: {m.trainable_params:,} params, "
              f"types={out['type'].tolist()[:3]}...")

    # Test gradient flow
    result['probabilities'].sum().backward()
    grad_count = sum(
        1 for p in arti.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for p in arti.parameters())
    print(f"\nGradient flow: {grad_count}/{total} params have gradients")

    # Test streaming
    print("\nStreaming test:")
    stream = StreamingARTI(arti)
    for i in range(5):
        emb = torch.randn(384)
        step_result = stream.step(embedding=emb)
        print(f"  Step {i}: {step_result['short_name']} ({step_result['confidence']:.0%})")

    traj = stream.get_trajectory()
    print(f"  Trajectory shape: {traj.shape}")
    print(f"  Type sequence: {stream.get_type_sequence()}")

    print("\nAll ARTI tests passed!")
