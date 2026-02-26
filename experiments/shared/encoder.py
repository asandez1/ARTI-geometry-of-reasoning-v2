#!/usr/bin/env python3
"""
Module 1: Semantic Encoder for CO-FRN (Section 3.1 of draft).

Frozen pretrained transformer -> mean-pool -> linear projection + LayerNorm + GELU.

Replaces Paper 11 E3's toy SemanticEncoder (learnable projection only) with
a proper frozen pretrained GPT-2 wrapper.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FrozenEncoder(nn.Module):
    """
    Frozen pretrained GPT-2 encoder with trainable projection head.

    Architecture:
        Input text tokens -> Frozen GPT-2 -> Token embeddings [B, L, D_model]
                          -> Mean pooling -> [B, D_model]
                          -> Linear(D_model, hidden_dim) + LayerNorm + GELU -> |s_0>

    The GPT-2 weights are never updated. Only the projection head is trainable.
    This isolates the factorization module's contribution.

    When unfreeze_last_n > 0, the last N transformer blocks + final layernorm
    are unfrozen for fine-tuning (E13).

    Supported models: gpt2 (768), gpt2-medium (1024), gpt2-large (1280), gpt2-xl (1600)
    """

    MODEL_DIMS = {
        'gpt2': 768,
        'gpt2-medium': 1024,
        'gpt2-large': 1280,
        'gpt2-xl': 1600,
    }

    def __init__(
        self,
        model_name: str = 'gpt2',
        hidden_dim: int = 256,
        load_pretrained: bool = True,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.model_dim = self.MODEL_DIMS.get(model_name, 768)
        self.unfreeze_last_n = unfreeze_last_n

        # Load frozen pretrained model
        if load_pretrained:
            from transformers import AutoModel, AutoTokenizer
            self.transformer = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Freeze all transformer parameters
            for param in self.transformer.parameters():
                param.requires_grad = False

            # Selectively unfreeze last N transformer blocks + final layernorm
            if unfreeze_last_n > 0 and hasattr(self.transformer, 'h'):
                total_blocks = len(self.transformer.h)
                start_idx = max(0, total_blocks - unfreeze_last_n)
                for block in self.transformer.h[start_idx:]:
                    for param in block.parameters():
                        param.requires_grad = True
                # Also unfreeze final layer norm
                if hasattr(self.transformer, 'ln_f'):
                    for param in self.transformer.ln_f.parameters():
                        param.requires_grad = True
        else:
            self.transformer = None
            self.tokenizer = None

        # Trainable projection head: D_model -> hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract mean-pooled embeddings from the transformer.

        When unfreeze_last_n > 0, gradients flow through unfrozen layers.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask (1=real, 0=pad)

        Returns:
            pooled: [B, D_model] mean-pooled token embeddings
        """
        ctx = contextlib.nullcontext() if self.unfreeze_last_n > 0 else torch.no_grad()
        with ctx:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state  # [B, L, D_model]

        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        return pooled  # [B, D_model]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text to initial state |s_0>.

        Accepts either raw token IDs (runs through frozen transformer)
        or pre-extracted embeddings (skips transformer, useful for caching).

        Args:
            input_ids: [B, L] token IDs (if providing raw text)
            attention_mask: [B, L] attention mask
            embeddings: [B, D_model] pre-extracted embeddings (bypasses transformer)

        Returns:
            s0: [B, hidden_dim] initial state
        """
        if embeddings is not None:
            pooled = embeddings
        elif input_ids is not None:
            pooled = self.get_embeddings(input_ids, attention_mask)
        else:
            raise ValueError("Provide either input_ids or embeddings")

        s0 = self.projection(pooled)  # [B, hidden_dim]
        return s0

    def tokenize(self, texts: list, max_length: int = 512) -> dict:
        """
        Tokenize a list of texts for the encoder.

        Args:
            texts: List of strings
            max_length: Maximum sequence length

        Returns:
            Dict with input_ids and attention_mask tensors
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )

    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters (projection + any unfrozen encoder layers)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        """Total parameters including frozen transformer."""
        return sum(p.numel() for p in self.parameters())


class SentenceTransformerEncoder(nn.Module):
    """
    Frozen sentence-transformer encoder with trainable projection head.

    Uses sentence-transformers library (already in requirements.txt) which
    produces embeddings tuned for semantic similarity — critical for cross-domain
    reasoning signal that GPT-2 misses (Paper 11 E4 used all-MiniLM-L6-v2).

    Architecture:
        Input texts -> Frozen SentenceTransformer -> [B, model_dim]
                    -> Linear(model_dim, hidden_dim) + LayerNorm + GELU -> |s_0>
    """

    MODELS = {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
        'intfloat/e5-small-v2': 384,
    }

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        hidden_dim: int = 256,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.model_dim = self.MODELS.get(model_name, 384)

        if load_pretrained:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name)
            # Freeze all backbone parameters
            for param in self._st_model.parameters():
                param.requires_grad = False
        else:
            self._st_model = None

        # Trainable projection head: model_dim -> hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(self.model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def encode_texts(self, texts: list, batch_size: int = 64) -> torch.Tensor:
        """
        Encode texts to raw embeddings (before projection).

        Args:
            texts: List of strings
            batch_size: Encoding batch size

        Returns:
            embeddings: [N, model_dim] tensor (detached, on CPU)
        """
        if self._st_model is None:
            raise ValueError("Pretrained model not loaded")
        with torch.no_grad():
            embs = self._st_model.encode(
                texts, batch_size=batch_size,
                show_progress_bar=False, convert_to_tensor=True,
            )
        return embs.cpu()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project pre-encoded embeddings through trainable head.

        Args:
            embeddings: [B, model_dim] raw sentence-transformer embeddings

        Returns:
            s0: [B, hidden_dim] initial state
        """
        return self.projection(embeddings)

    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters (projection head only)."""
        return sum(p.numel() for p in self.projection.parameters())

    @property
    def total_params(self) -> int:
        """Total parameters including frozen transformer."""
        return sum(p.numel() for p in self.parameters())


class PrecomputedEncoder(nn.Module):
    """
    Lightweight encoder for pre-extracted embeddings.

    When embeddings are cached (recommended for efficiency), this module
    just applies the trainable projection without loading the transformer.
    Useful for fast iteration during experiment development.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project pre-extracted embeddings to hidden_dim.

        Args:
            embeddings: [B, input_dim] pre-extracted encoder embeddings

        Returns:
            s0: [B, hidden_dim] initial state
        """
        return self.projection(embeddings)

    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters (projection head only)."""
        return sum(p.numel() for p in self.projection.parameters())


if __name__ == "__main__":
    print("Testing encoders...")

    # Test PrecomputedEncoder (no model download needed)
    enc = PrecomputedEncoder(input_dim=768, hidden_dim=256)
    x = torch.randn(4, 768)
    s0 = enc(x)
    print(f"PrecomputedEncoder: {x.shape} -> {s0.shape}")
    print(f"  Trainable params: {sum(p.numel() for p in enc.parameters()):,}")

    # Test FrozenEncoder with pre-extracted embeddings (no model download)
    frozen = FrozenEncoder(model_name='gpt2', hidden_dim=256, load_pretrained=False)
    s0_frozen = frozen(embeddings=x)
    print(f"\nFrozenEncoder (embeddings mode): {x.shape} -> {s0_frozen.shape}")
    print(f"  Projection params: {frozen.trainable_params:,}")

    # Gradient check
    s0.sum().backward()
    grad_params = sum(1 for p in enc.parameters() if p.grad is not None)
    total = sum(1 for _ in enc.parameters())
    print(f"\nGradient flow: {grad_params}/{total} params have gradients")

    print("\nEncoder tests passed!")
