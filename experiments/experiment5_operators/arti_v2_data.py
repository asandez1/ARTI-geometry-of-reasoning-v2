#!/usr/bin/env python3
"""
Data pipeline for ARTI v2 trajectory-based classification.

Loads v1 dataset.pt to get texts and labels, then:
  1. Segments each text into clauses via segment_text()
  2. Encodes each clause with the frozen sentence-transformer
  3. Stores variable-length clause embedding sequences + labels

The TrajectoryDataset stores pre-computed clause embeddings as a list
of [n_clauses, 384] tensors (variable length per sample).
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.text_utils import segment_text
from shared.reasoning_types import (
    ReasoningType, N_REASONING_TYPES, REASONING_TYPES,
)

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    Dataset of reasoning traces stored as variable-length clause embeddings.

    Each sample has:
    - clause_embeddings: [n_clauses, encoder_dim] tensor (variable n_clauses)
    - label: int (ReasoningType enum value)
    - text: str (original text)
    - n_clauses: int (number of clauses)
    """

    def __init__(
        self,
        clause_embeddings_list: List[torch.Tensor],
        labels: torch.Tensor,
        texts: Optional[List[str]] = None,
    ):
        self.clause_embeddings_list = clause_embeddings_list
        self.labels = labels
        self.texts = texts or [""] * len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'clause_embeddings': self.clause_embeddings_list[idx],
            'label': self.labels[idx],
        }

    def split(
        self, train_ratio: float = 0.8, seed: int = 42
    ) -> Tuple['TrajectoryDataset', 'TrajectoryDataset']:
        """Split into train/val with stratified sampling (matching v1 split)."""
        rng = np.random.RandomState(seed)
        n = len(self)
        indices = rng.permutation(n)

        train_indices = []
        val_indices = []
        for label_val in range(N_REASONING_TYPES):
            label_mask = (self.labels == label_val).numpy()
            label_indices = indices[label_mask[indices]]
            split_point = int(len(label_indices) * train_ratio)
            train_indices.extend(label_indices[:split_point].tolist())
            val_indices.extend(label_indices[split_point:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        def subset(idxs):
            return TrajectoryDataset(
                clause_embeddings_list=[self.clause_embeddings_list[i] for i in idxs],
                labels=self.labels[idxs],
                texts=[self.texts[i] for i in idxs],
            )

        return subset(train_indices), subset(val_indices)


def collate_trajectory(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length clause embedding sequences.

    Returns:
        Dict with:
        - clause_embeddings_list: list of [n_clauses_i, encoder_dim] tensors
        - labels: [B] tensor
    """
    return {
        'clause_embeddings_list': [b['clause_embeddings'] for b in batch],
        'labels': torch.stack([b['label'] for b in batch]),
    }


def build_trajectory_dataset(
    v1_dataset_path: str,
    encoder,
    batch_size: int = 64,
) -> TrajectoryDataset:
    """
    Build trajectory dataset from v1's saved dataset.

    For each text in the v1 dataset:
      1. segment_text() -> list of clauses
      2. Encode each clause with the frozen encoder
      3. Store as [n_clauses, encoder_dim] tensor

    Args:
        v1_dataset_path: path to v1's dataset.pt
        encoder: SentenceTransformerEncoder instance
        batch_size: encoding batch size

    Returns:
        TrajectoryDataset with pre-computed clause embeddings
    """
    logger.info(f"Loading v1 dataset from {v1_dataset_path}")
    data = torch.load(v1_dataset_path, weights_only=False)
    texts = data.get('texts', [])
    labels = data['labels']

    if not texts:
        raise ValueError("v1 dataset has no texts stored")

    logger.info(f"  {len(texts)} samples loaded")

    # Segment all texts
    logger.info("Segmenting texts into clauses...")
    all_segments = []
    clause_counts = []
    for text in texts:
        segs = segment_text(text)
        all_segments.append(segs)
        clause_counts.append(len(segs))

    clause_counts = np.array(clause_counts)
    logger.info(f"  Clause counts: mean={clause_counts.mean():.1f}, "
                f"min={clause_counts.min()}, max={clause_counts.max()}, "
                f"median={np.median(clause_counts):.0f}")

    # Flatten all clauses for batch encoding
    flat_clauses = []
    sample_boundaries = [0]
    for segs in all_segments:
        flat_clauses.extend(segs)
        sample_boundaries.append(len(flat_clauses))

    logger.info(f"  Total clauses to encode: {len(flat_clauses)}")

    # Encode all clauses in batches
    logger.info("Encoding clauses with frozen encoder...")
    all_embs = []
    for i in range(0, len(flat_clauses), batch_size):
        batch_texts = flat_clauses[i:i + batch_size]
        with torch.no_grad():
            embs = encoder.encode_texts(batch_texts)  # [batch, 384]
        all_embs.append(embs)
    all_embs = torch.cat(all_embs, dim=0)  # [total_clauses, 384]

    # Split back into per-sample variable-length tensors
    clause_embeddings_list = []
    for i in range(len(texts)):
        start = sample_boundaries[i]
        end = sample_boundaries[i + 1]
        clause_embeddings_list.append(all_embs[start:end])  # [n_clauses_i, 384]

    logger.info(f"  Built {len(clause_embeddings_list)} trajectory samples")

    return TrajectoryDataset(
        clause_embeddings_list=clause_embeddings_list,
        labels=labels,
        texts=texts,
    )


def save_trajectory_dataset(dataset: TrajectoryDataset, path: str):
    """Save trajectory dataset to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'clause_embeddings_list': dataset.clause_embeddings_list,
        'labels': dataset.labels,
        'texts': dataset.texts,
    }, path)
    logger.info(f"Trajectory dataset saved to {path} ({len(dataset)} samples)")


def load_trajectory_dataset(path: str) -> TrajectoryDataset:
    """Load trajectory dataset from disk."""
    data = torch.load(path, weights_only=False)
    return TrajectoryDataset(
        clause_embeddings_list=data['clause_embeddings_list'],
        labels=data['labels'],
        texts=data.get('texts', []),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing ARTI v2 data pipeline...")

    # Test segment_text
    from shared.text_utils import segment_text
    tests = [
        "The temperature dropped, so the pipes burst.",
        "All mammals are warm-blooded. A whale is a mammal. Therefore warm-blooded.",
    ]
    for t in tests:
        segs = segment_text(t)
        print(f"  [{len(segs)} segs] {t[:50]}...")

    # Test collate
    dummy_batch = [
        {'clause_embeddings': torch.randn(3, 384), 'label': torch.tensor(0)},
        {'clause_embeddings': torch.randn(5, 384), 'label': torch.tensor(1)},
    ]
    collated = collate_trajectory(dummy_batch)
    print(f"\nCollate test: {len(collated['clause_embeddings_list'])} samples, "
          f"labels shape: {collated['labels'].shape}")

    print("\nAll arti_v2_data tests passed!")
