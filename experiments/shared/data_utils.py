#!/usr/bin/env python3
"""
Dataset Loaders for CO-FRN Experiments.

Loads and preprocesses benchmarks for the four experiments:
- GSM8K: Math word problems (multi-step)
- ARC Challenge: Science multiple-choice QA (single-step)
- StrategyQA: Commonsense yes/no (multi-step)
- FOLIO: Formal logic NLI (single-step)
- SVAMP: Math word problems (near-domain transfer target)
- OpenBookQA: Science QA (near-domain transfer target)

All datasets are loaded via HuggingFace datasets library and return
pre-tokenized / pre-embedded tensors for efficient training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
import inspect
import re

logger = logging.getLogger(__name__)


# ─── Dataset Configurations ────────────────────────────────────────────────

BENCHMARK_CONFIGS = {
    'gsm8k': {
        'hf_name': 'openai/gsm8k',
        'hf_subset': 'main',
        'task_type': 'multi_step',
        'n_classes': None,  # open-ended, converted to multiple-choice
        'domain': 'math',
    },
    'arc_challenge': {
        'hf_name': 'allenai/ai2_arc',
        'hf_subset': 'ARC-Challenge',
        'task_type': 'single_step',
        'n_classes': 4,
        'domain': 'science',
    },
    'strategyqa': {
        'hf_name': 'ChilleD/StrategyQA',
        'hf_subset': None,
        'task_type': 'multi_step',
        'n_classes': 2,
        'domain': 'commonsense',
    },
    'folio': {
        'hf_name': 'tasksource/folio',
        'hf_subset': None,
        'task_type': 'single_step',
        'n_classes': 3,  # entailment, contradiction, neutral
        'domain': 'logic',
    },
    'svamp': {
        'hf_name': 'ChilleD/SVAMP',
        'hf_subset': None,
        'task_type': 'multi_step',
        'n_classes': None,
        'domain': 'math',
    },
    'openbookqa': {
        'hf_name': 'allenai/openbookqa',
        'hf_subset': 'main',
        'task_type': 'single_step',
        'n_classes': 4,
        'domain': 'science',
    },
}


# ─── Embedding Cache ───────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Cache pre-extracted GPT-2 embeddings for all dataset texts.

    Running the frozen encoder once and caching the outputs makes
    training ~10x faster since we skip the transformer forward pass.
    """

    def __init__(self, encoder=None, device: str = 'cpu'):
        self.cache = {}
        self.encoder = encoder
        self.device = device

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Encode a list of texts, using cache where available.

        Args:
            texts: list of strings
            batch_size: encoding batch size
            max_length: max token length

        Returns:
            embeddings: [N, D_model] tensor
        """
        if self.encoder is None:
            raise ValueError("Encoder required for embedding texts")

        # Check cache
        uncached_indices = []
        uncached_texts = []
        for i, t in enumerate(texts):
            if t not in self.cache:
                uncached_indices.append(i)
                uncached_texts.append(t)

        # Encode uncached
        if uncached_texts:
            all_embs = []
            for start in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[start:start + batch_size]
                tokens = self.encoder.tokenize(batch_texts, max_length=max_length)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                with torch.no_grad():
                    embs = self.encoder.get_embeddings(
                        tokens['input_ids'], tokens['attention_mask']
                    )
                all_embs.append(embs.cpu())

            all_embs = torch.cat(all_embs, dim=0)
            for i, text in enumerate(uncached_texts):
                self.cache[text] = all_embs[i]

        # Collect results
        result = torch.stack([self.cache[t] for t in texts])
        return result


# ─── Base Dataset ──────────────────────────────────────────────────────────

class BenchmarkDataset(Dataset):
    """
    Base dataset class for CO-FRN benchmarks.

    Stores pre-embedded questions, answer choices, and labels.
    All datasets are converted to multiple-choice format for
    uniform handling across experiments.
    """

    def __init__(
        self,
        question_embeddings: torch.Tensor,
        answer_embeddings: torch.Tensor,
        labels: torch.Tensor,
        domain: str = 'unknown',
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Args:
            question_embeddings: [N, D_model] question/premise embeddings
            answer_embeddings: [N, n_choices, D_model] answer choice embeddings
            labels: [N] correct answer indices
            domain: domain name for transfer experiments
            metadata: optional per-example metadata
        """
        self.question_embeddings = question_embeddings
        self.answer_embeddings = answer_embeddings
        self.labels = labels
        self.domain = domain
        self.metadata = metadata

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'embeddings': self.question_embeddings[idx],
            'answer_embeddings': self.answer_embeddings[idx],
            'labels': self.labels[idx],
        }
        if self.metadata is not None:
            item['metadata'] = self.metadata[idx]
        return item

    def subset(self, indices) -> 'BenchmarkDataset':
        """Create a subset with given indices."""
        return BenchmarkDataset(
            question_embeddings=self.question_embeddings[indices],
            answer_embeddings=self.answer_embeddings[indices],
            labels=self.labels[indices],
            domain=self.domain,
            metadata=[self.metadata[i] for i in indices] if self.metadata else None,
        )


def collate_benchmark(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for BenchmarkDataset."""
    return {
        'embeddings': torch.stack([b['embeddings'] for b in batch]),
        'answer_embeddings': torch.stack([b['answer_embeddings'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


# ─── Text-Based Dataset (for unfrozen encoder training) ───────────────────

class TextBenchmarkDataset(Dataset):
    """
    Dataset that stores raw text + tokenized inputs for live encoding.

    Used with unfrozen encoder (E13) where gradients must flow through
    the transformer — cannot use pre-computed embeddings.
    """

    def __init__(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        domain: str = 'unknown',
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Args:
            question_input_ids: [N, L] tokenized questions
            question_attention_mask: [N, L] question masks
            answer_input_ids: [N, n_choices, L] tokenized answers
            answer_attention_mask: [N, n_choices, L] answer masks
            labels: [N] correct answer indices
            domain: domain name
            metadata: optional per-example metadata
        """
        self.question_input_ids = question_input_ids
        self.question_attention_mask = question_attention_mask
        self.answer_input_ids = answer_input_ids
        self.answer_attention_mask = answer_attention_mask
        self.labels = labels
        self.domain = domain
        self.metadata = metadata

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.question_input_ids[idx],
            'attention_mask': self.question_attention_mask[idx],
            'answer_input_ids': self.answer_input_ids[idx],
            'answer_attention_mask': self.answer_attention_mask[idx],
            'labels': self.labels[idx],
        }

    def subset(self, indices) -> 'TextBenchmarkDataset':
        """Create a subset with given indices."""
        return TextBenchmarkDataset(
            question_input_ids=self.question_input_ids[indices],
            question_attention_mask=self.question_attention_mask[indices],
            answer_input_ids=self.answer_input_ids[indices],
            answer_attention_mask=self.answer_attention_mask[indices],
            labels=self.labels[indices],
            domain=self.domain,
            metadata=[self.metadata[i] for i in indices] if self.metadata else None,
        )


def collate_text_benchmark(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for TextBenchmarkDataset."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'answer_input_ids': torch.stack([b['answer_input_ids'] for b in batch]),
        'answer_attention_mask': torch.stack([b['answer_attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


def load_text_benchmark(
    benchmark: str,
    split: str = 'train',
    tokenizer=None,
    max_length: int = 128,
    seed: int = 42,
) -> TextBenchmarkDataset:
    """
    Load a benchmark dataset with tokenized text (for unfrozen encoder).

    Args:
        benchmark: one of gsm8k, arc_challenge, strategyqa, folio
        split: dataset split
        tokenizer: HuggingFace tokenizer (from encoder.tokenizer)
        max_length: max token length
        seed: random seed

    Returns:
        TextBenchmarkDataset with tokenized questions and answers
    """
    if benchmark not in LOADERS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if tokenizer is None:
        raise ValueError("tokenizer required for text-based loading")

    loader_fn = LOADERS[benchmark]
    kwargs = {'split': split}
    if 'seed' in inspect.signature(loader_fn).parameters:
        kwargs['seed'] = seed
    questions, choices, labels = loader_fn(**kwargs)

    logger.info(f"Tokenizing {benchmark}/{split}: {len(questions)} examples")

    # Tokenize questions
    q_tok = tokenizer(
        questions, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt',
    )

    # Tokenize answers: flatten, tokenize, reshape
    n_choices = len(choices[0])
    all_choice_texts = []
    for cs in choices:
        all_choice_texts.extend(cs)

    a_tok = tokenizer(
        all_choice_texts, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt',
    )

    N = len(questions)
    a_input_ids = a_tok['input_ids'].reshape(N, n_choices, max_length)
    a_attention_mask = a_tok['attention_mask'].reshape(N, n_choices, max_length)

    config = BENCHMARK_CONFIGS[benchmark]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TextBenchmarkDataset(
        question_input_ids=q_tok['input_ids'],
        question_attention_mask=q_tok['attention_mask'],
        answer_input_ids=a_input_ids,
        answer_attention_mask=a_attention_mask,
        labels=labels_tensor,
        domain=config['domain'],
        metadata=[{'question': q, 'choices': c} for q, c in zip(questions, choices)],
    )


# ─── Dataset Loaders ──────────────────────────────────────────────────────

def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numerical answer from GSM8K answer string."""
    # GSM8K format: "...#### 42"
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip().replace(',', '')
    return answer_text.strip()


def load_gsm8k(
    split: str = 'train',
    n_choices: int = 4,
    seed: int = 42,
) -> Tuple[List[str], List[List[str]], List[int]]:
    """
    Load GSM8K and convert to multiple-choice format.

    Generates distractors by perturbing the correct numerical answer.

    Returns:
        questions: list of question strings
        choices: list of [n_choices] answer strings per question
        labels: list of correct answer indices
    """
    from datasets import load_dataset

    ds = load_dataset('openai/gsm8k', 'main', split=split)
    rng = np.random.RandomState(seed)

    questions = []
    choices = []
    labels = []

    for item in ds:
        question = item['question']
        answer = _extract_gsm8k_answer(item['answer'])

        # Generate numerical distractors
        try:
            correct_val = float(answer)
            distractors = set()
            attempts = 0
            while len(distractors) < n_choices - 1 and attempts < 100:
                # Random perturbation strategies
                strategy = rng.choice(['add', 'mult', 'digit'])
                if strategy == 'add':
                    delta = rng.choice([-2, -1, 1, 2, 5, 10]) * max(1, abs(correct_val) * 0.1)
                    d = correct_val + delta
                elif strategy == 'mult':
                    factor = rng.choice([0.5, 0.75, 1.5, 2.0])
                    d = correct_val * factor
                else:
                    d = correct_val + rng.randint(-10, 10)

                d_str = str(int(d)) if d == int(d) else f"{d:.2f}"
                if d_str != answer and d_str not in distractors:
                    distractors.add(d_str)
                attempts += 1

            # Fill remaining if needed
            while len(distractors) < n_choices - 1:
                distractors.add(str(int(correct_val + len(distractors) + 1)))

        except ValueError:
            distractors = {f"answer_{i}" for i in range(n_choices - 1)}

        all_choices = [answer] + list(distractors)[:n_choices - 1]
        correct_idx = 0

        # Shuffle
        perm = rng.permutation(len(all_choices))
        all_choices = [all_choices[i] for i in perm]
        correct_idx = int(np.where(perm == 0)[0][0])

        questions.append(question)
        choices.append(all_choices)
        labels.append(correct_idx)

    return questions, choices, labels


def load_arc_challenge(
    split: str = 'train',
) -> Tuple[List[str], List[List[str]], List[int]]:
    """Load ARC Challenge dataset."""
    from datasets import load_dataset

    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split=split)

    questions = []
    choices = []
    labels = []

    for item in ds:
        questions.append(item['question'])
        choice_texts = item['choices']['text']

        # Pad to 4 choices if needed
        while len(choice_texts) < 4:
            choice_texts.append("[N/A]")

        choices.append(choice_texts[:4])

        # Convert letter label to index
        answer_key = item['answerKey']
        if answer_key in 'ABCDE':
            label_idx = ord(answer_key) - ord('A')
        else:
            label_idx = int(answer_key) - 1
        labels.append(min(label_idx, 3))

    return questions, choices, labels


def load_strategyqa(
    split: str = 'train',
) -> Tuple[List[str], List[List[str]], List[int]]:
    """Load StrategyQA (yes/no questions)."""
    from datasets import load_dataset

    ds = load_dataset('ChilleD/StrategyQA', split=split)

    questions = []
    choices = []
    labels = []

    for item in ds:
        questions.append(item['question'])
        choices.append(['Yes', 'No'])
        labels.append(0 if item['answer'] else 1)

    return questions, choices, labels


def load_folio(
    split: str = 'train',
) -> Tuple[List[str], List[List[str]], List[int]]:
    """Load FOLIO (formal logic NLI)."""
    from datasets import load_dataset

    ds = load_dataset('tasksource/folio', split=split if split != 'test' else 'validation')

    label_map = {'True': 0, 'False': 1, 'Uncertain': 2}

    questions = []
    choices = []
    labels = []

    for item in ds:
        premise = item.get('premises', item.get('premise', ''))
        conclusion = item.get('conclusion', '')
        text = f"Premises: {premise}\nConclusion: {conclusion}"

        questions.append(text)
        choices.append(['Entailment', 'Contradiction', 'Neutral'])

        label_str = item.get('label', 'Uncertain')
        labels.append(label_map.get(str(label_str), 2))

    return questions, choices, labels


def load_svamp(
    split: str = 'train',
    n_choices: int = 4,
    seed: int = 42,
) -> Tuple[List[str], List[List[str]], List[int]]:
    """Load SVAMP math dataset."""
    from datasets import load_dataset

    ds = load_dataset('ChilleD/SVAMP', split=split if split != 'test' else 'test')
    rng = np.random.RandomState(seed)

    questions = []
    choices = []
    labels = []

    for item in ds:
        body = item.get('Body', item.get('body', ''))
        question = item.get('Question', item.get('question', ''))
        text = f"{body} {question}"
        answer = str(item.get('Answer', item.get('answer', '0')))

        try:
            correct_val = float(answer)
            distractors = set()
            for _ in range(50):
                delta = rng.choice([-5, -3, -1, 1, 3, 5]) + rng.randn() * max(1, abs(correct_val) * 0.2)
                d = correct_val + delta
                d_str = str(int(d)) if d == int(d) else f"{d:.1f}"
                if d_str != answer:
                    distractors.add(d_str)
                if len(distractors) >= n_choices - 1:
                    break
            while len(distractors) < n_choices - 1:
                distractors.add(str(int(correct_val + len(distractors) + 1)))
        except ValueError:
            distractors = {f"ans_{i}" for i in range(n_choices - 1)}

        all_choices = [answer] + list(distractors)[:n_choices - 1]
        perm = rng.permutation(len(all_choices))
        all_choices = [all_choices[i] for i in perm]
        correct_idx = int(np.where(perm == 0)[0][0])

        questions.append(text)
        choices.append(all_choices)
        labels.append(correct_idx)

    return questions, choices, labels


def load_openbookqa(
    split: str = 'train',
) -> Tuple[List[str], List[List[str]], List[int]]:
    """Load OpenBookQA dataset."""
    from datasets import load_dataset

    ds = load_dataset('allenai/openbookqa', 'main', split=split)

    questions = []
    choices = []
    labels = []

    for item in ds:
        questions.append(item['question_stem'])
        choice_texts = item['choices']['text']
        while len(choice_texts) < 4:
            choice_texts.append("[N/A]")
        choices.append(choice_texts[:4])

        answer_key = item['answerKey']
        label_idx = ord(answer_key) - ord('A')
        labels.append(min(label_idx, 3))

    return questions, choices, labels


# ─── Dataset Loading Map ───────────────────────────────────────────────────

LOADERS = {
    'gsm8k': load_gsm8k,
    'arc_challenge': load_arc_challenge,
    'strategyqa': load_strategyqa,
    'folio': load_folio,
    'svamp': load_svamp,
    'openbookqa': load_openbookqa,
}


# ─── High-Level API ───────────────────────────────────────────────────────

def load_benchmark(
    benchmark: str,
    split: str = 'train',
    encoder=None,
    cache: Optional[EmbeddingCache] = None,
    hidden_dim: int = 256,
    max_length: int = 512,
    device: str = 'cpu',
    seed: int = 42,
) -> BenchmarkDataset:
    """
    Load a benchmark dataset with pre-computed embeddings.

    Args:
        benchmark: one of gsm8k, arc_challenge, strategyqa, folio, svamp, openbookqa
        split: train, validation, or test
        encoder: FrozenEncoder instance (for embedding)
        cache: optional EmbeddingCache
        hidden_dim: projection dimension
        max_length: max token length for encoding
        device: torch device
        seed: random seed

    Returns:
        BenchmarkDataset with pre-embedded questions and answers
    """
    if benchmark not in LOADERS:
        raise ValueError(f"Unknown benchmark: {benchmark}. "
                         f"Available: {list(LOADERS.keys())}")

    config = BENCHMARK_CONFIGS[benchmark]
    loader_fn = LOADERS[benchmark]

    # Load raw data
    kwargs = {'split': split}
    if 'seed' in inspect.signature(loader_fn).parameters:
        kwargs['seed'] = seed
    questions, choices, labels = loader_fn(**kwargs)

    logger.info(f"Loaded {benchmark}/{split}: {len(questions)} examples")

    # Embed questions and choices
    if encoder is not None:
        if cache is None:
            cache = EmbeddingCache(encoder=encoder, device=device)

        # Embed questions
        q_embs = cache.encode_texts(questions, max_length=max_length)

        # Embed all choice texts
        all_choice_texts = []
        for cs in choices:
            all_choice_texts.extend(cs)
        c_embs = cache.encode_texts(all_choice_texts, max_length=max_length)

        # Reshape choice embeddings
        n_choices = len(choices[0])
        c_embs = c_embs.reshape(len(questions), n_choices, -1)

        # Project through encoder projection to get hidden_dim
        with torch.no_grad():
            q_projected = encoder.projection(q_embs.to(device)).cpu()
            c_flat = c_embs.reshape(-1, c_embs.shape[-1])
            c_projected = encoder.projection(c_flat.to(device)).cpu()
            c_projected = c_projected.reshape(len(questions), n_choices, -1)
    else:
        # Generate random embeddings (for testing without model download)
        logger.warning(f"No encoder provided — using random embeddings for {benchmark}")
        q_projected = torch.randn(len(questions), hidden_dim)
        n_choices = len(choices[0])
        c_projected = torch.randn(len(questions), n_choices, hidden_dim)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return BenchmarkDataset(
        question_embeddings=q_projected,
        answer_embeddings=c_projected,
        labels=labels_tensor,
        domain=config['domain'],
        metadata=[{'question': q, 'choices': c} for q, c in zip(questions, choices)],
    )


def load_benchmark_splits(
    benchmark: str,
    encoder=None,
    device: str = 'cpu',
    seed: int = 42,
    hidden_dim: int = 256,
) -> Dict[str, BenchmarkDataset]:
    """
    Load train/val/test splits for a benchmark.

    Returns:
        Dict with 'train', 'validation', 'test' keys
    """
    cache = EmbeddingCache(encoder=encoder, device=device) if encoder else None

    splits = {}
    split_names = {
        'gsm8k': ['train', 'test'],
        'arc_challenge': ['train', 'validation', 'test'],
        'strategyqa': ['train', 'test'],
        'folio': ['train', 'validation'],
        'svamp': ['train', 'test'],
        'openbookqa': ['train', 'validation', 'test'],
    }

    for split in split_names.get(benchmark, ['train']):
        try:
            ds = load_benchmark(
                benchmark, split=split, encoder=encoder,
                cache=cache, hidden_dim=hidden_dim, device=device, seed=seed,
            )
            splits[split] = ds
        except Exception as e:
            logger.warning(f"Could not load {benchmark}/{split}: {e}")

    # Create validation from train if not available
    if 'validation' not in splits and 'train' in splits:
        train_ds = splits['train']
        n = len(train_ds)
        val_size = min(500, n // 5)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        splits['validation'] = train_ds.subset(indices[:val_size])
        splits['train'] = train_ds.subset(indices[val_size:])

    return splits


# ─── Sentence-Transformer Precomputation ──────────────────────────────

def precompute_embeddings_st(
    encoder_name: str,
    benchmark: str,
    split: str = 'train',
    hidden_dim: int = 256,
    seed: int = 42,
) -> BenchmarkDataset:
    """
    Encode benchmark with a sentence-transformer, return BenchmarkDataset.

    Uses SentenceTransformerEncoder.encode_texts() for questions and answers,
    stores raw embeddings (pre-projection). The BenchmarkDataset's
    question_embeddings are at model_dim (not hidden_dim) — the trainable
    projection is applied inside the model.

    Args:
        encoder_name: sentence-transformer model name
        benchmark: one of BENCHMARK_CONFIGS keys
        split: dataset split
        hidden_dim: not used for raw embeddings but stored for reference
        seed: random seed

    Returns:
        BenchmarkDataset with raw (pre-projection) embeddings
    """
    from .encoder import SentenceTransformerEncoder

    config = BENCHMARK_CONFIGS[benchmark]
    loader_fn = LOADERS[benchmark]

    # Load raw data
    kwargs = {'split': split}
    if 'seed' in inspect.signature(loader_fn).parameters:
        kwargs['seed'] = seed
    questions, choices, labels = loader_fn(**kwargs)

    logger.info(f"Precomputing ST embeddings for {benchmark}/{split}: "
                f"{len(questions)} examples with {encoder_name}")

    # Create encoder for embedding (no projection needed for raw)
    st_enc = SentenceTransformerEncoder(
        model_name=encoder_name, hidden_dim=hidden_dim, load_pretrained=True,
    )

    # Embed questions
    q_embs = st_enc.encode_texts(questions, batch_size=64)  # [N, model_dim]

    # Embed all choice texts
    all_choice_texts = []
    for cs in choices:
        all_choice_texts.extend(cs)
    c_embs = st_enc.encode_texts(all_choice_texts, batch_size=64)  # [N*C, model_dim]

    # Reshape choice embeddings
    n_choices = len(choices[0])
    c_embs = c_embs.reshape(len(questions), n_choices, -1)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Clean up to free memory
    del st_enc

    return BenchmarkDataset(
        question_embeddings=q_embs,
        answer_embeddings=c_embs,
        labels=labels_tensor,
        domain=config['domain'],
        metadata=[{'question': q, 'choices': c} for q, c in zip(questions, choices)],
    )


# ─── Mixed Domain Dataset ────────────────────────────────────────────

class MixedDomainDataset(Dataset):
    """
    Concatenates multiple BenchmarkDatasets, padding answer choices to max,
    tracking domain labels per example.

    Handles different n_choices across benchmarks by padding answer_embeddings
    to max(n_choices) with zero vectors and providing a valid_choices mask.
    """

    def __init__(self, datasets: Dict[str, BenchmarkDataset]):
        """
        Args:
            datasets: {name: BenchmarkDataset} mapping
        """
        self.domain_names = list(datasets.keys())

        # Find max n_choices across all datasets
        max_choices = max(ds.answer_embeddings.shape[1] for ds in datasets.values())
        emb_dim = next(iter(datasets.values())).question_embeddings.shape[1]

        all_q = []
        all_a = []
        all_labels = []
        all_domains = []
        all_valid = []

        for name, ds in datasets.items():
            n = len(ds)
            n_choices = ds.answer_embeddings.shape[1]

            all_q.append(ds.question_embeddings)
            all_labels.append(ds.labels)
            all_domains.extend([name] * n)

            # Pad answer embeddings if needed
            if n_choices < max_choices:
                pad = torch.zeros(n, max_choices - n_choices, emb_dim)
                all_a.append(torch.cat([ds.answer_embeddings, pad], dim=1))
                # Valid choices mask
                valid = torch.zeros(n, max_choices, dtype=torch.bool)
                valid[:, :n_choices] = True
                all_valid.append(valid)
            else:
                all_a.append(ds.answer_embeddings)
                all_valid.append(torch.ones(n, max_choices, dtype=torch.bool))

        self.question_embeddings = torch.cat(all_q, dim=0)
        self.answer_embeddings = torch.cat(all_a, dim=0)
        self.labels = torch.cat(all_labels, dim=0)
        self.domains = all_domains
        self.valid_choices = torch.cat(all_valid, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embeddings': self.question_embeddings[idx],
            'answer_embeddings': self.answer_embeddings[idx],
            'labels': self.labels[idx],
            'valid_choices': self.valid_choices[idx],
            'domain': self.domains[idx],
        }


def collate_mixed_domain(batch: List[Dict]) -> Dict:
    """Collate function for MixedDomainDataset."""
    return {
        'embeddings': torch.stack([b['embeddings'] for b in batch]),
        'answer_embeddings': torch.stack([b['answer_embeddings'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'valid_choices': torch.stack([b['valid_choices'] for b in batch]),
        'domains': [b['domain'] for b in batch],
    }


if __name__ == "__main__":
    print("Testing data_utils (random embeddings mode)...")

    # Test each loader without downloading models
    for name in ['gsm8k', 'arc_challenge', 'strategyqa', 'folio', 'svamp', 'openbookqa']:
        try:
            loader_fn = LOADERS[name]
            kwargs = {'split': 'train'}
            if 'seed' in inspect.signature(loader_fn).parameters:
                kwargs['seed'] = 42
            questions, choices, labels = loader_fn(**kwargs)
            print(f"\n{name}: {len(questions)} examples, "
                  f"{len(choices[0])} choices/example")
            print(f"  Sample Q: {questions[0][:80]}...")
            print(f"  Choices: {choices[0]}")
            print(f"  Label: {labels[0]}")
        except Exception as e:
            print(f"\n{name}: SKIPPED ({e})")

    # Test BenchmarkDataset with random embeddings
    print("\n--- Testing BenchmarkDataset ---")
    ds = BenchmarkDataset(
        question_embeddings=torch.randn(100, 256),
        answer_embeddings=torch.randn(100, 4, 256),
        labels=torch.randint(0, 4, (100,)),
        domain='test',
    )
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_benchmark)
    batch = next(iter(loader))
    print(f"  Batch embeddings: {batch['embeddings'].shape}")
    print(f"  Batch answers: {batch['answer_embeddings'].shape}")
    print(f"  Batch labels: {batch['labels'].shape}")

    print("\ndata_utils tests passed!")
