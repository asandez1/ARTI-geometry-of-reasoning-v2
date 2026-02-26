# Paper 12: Factorized Reasoning Networks — Shared Modules
# CO-FRN (Continuous Operator Factorized Reasoning Network)

from .encoder import FrozenEncoder, PrecomputedEncoder, SentenceTransformerEncoder
from .operators import StructuralOperator, ContinuousOperatorManifold, SimpleMLPOperator
from .factorization import FactorizationLayer, ContinuousFactorizationModule
from .reasoning_engine import HilbertTreeFast, DirectScorer, ARTIRoutedScorer, ReasoningEngine
from .model import COFRN, COFRNConfig
from .baselines import LinearProbe, MLPHead, LoRAModel, FullFineTune
from .train_utils import (
    TrainConfig, CurriculumPhase, Trainer,
    train_model, evaluate_model, curriculum_train,
    measure_inference_latency, save_results,
)
from .data_utils import (
    precompute_embeddings_st, MixedDomainDataset, collate_mixed_domain,
)
from .metrics import (
    compute_centered_redundancy, bootstrap_rho_ci,
    compute_pareto_frontier, ParetoPoint,
    transfer_ratio, sample_efficiency_ratio, adaptation_curve_auc,
    log_linear_fit, detect_saturation, bootstrap_slope_ci,
    anchor_utilization_entropy, per_domain_conditional_entropy,
    cross_domain_js_divergence,
    paired_permutation_test, cohens_d,
)
from .reasoning_types import (
    ReasoningType, N_REASONING_TYPES, REASONING_TYPES,
    TYPE_NAMES, TYPE_SHORT_NAMES, TYPE_COLORS,
    HeuristicLabeler,
)
from .controller import (
    GeometricReasoningController, ControllerConfig, TypeClassifier,
    TYPE_MERGE_MAP, CORE_TYPE_NAMES, N_CORE_TYPES, merge_labels,
)
from .arti import ARTI, ARTIConfig, StreamingARTI
from .arti_v2 import ARTIV2, ARTIV2Config, TrajectoryFeatureExtractor
from .text_utils import segment_text

__all__ = [
    # Encoder
    'FrozenEncoder', 'PrecomputedEncoder', 'SentenceTransformerEncoder',
    # Operators
    'StructuralOperator', 'ContinuousOperatorManifold', 'SimpleMLPOperator',
    # Factorization
    'FactorizationLayer', 'ContinuousFactorizationModule',
    # Reasoning
    'HilbertTreeFast', 'DirectScorer', 'ARTIRoutedScorer', 'ReasoningEngine',
    # Model
    'COFRN', 'COFRNConfig',
    # Baselines
    'LinearProbe', 'MLPHead', 'LoRAModel', 'FullFineTune',
    # Training
    'TrainConfig', 'CurriculumPhase', 'Trainer',
    'train_model', 'evaluate_model', 'curriculum_train',
    'measure_inference_latency', 'save_results',
    # Data utils
    'precompute_embeddings_st', 'MixedDomainDataset', 'collate_mixed_domain',
    # Metrics
    'compute_centered_redundancy', 'bootstrap_rho_ci',
    'compute_pareto_frontier', 'ParetoPoint',
    'transfer_ratio', 'sample_efficiency_ratio', 'adaptation_curve_auc',
    'log_linear_fit', 'detect_saturation', 'bootstrap_slope_ci',
    'anchor_utilization_entropy', 'per_domain_conditional_entropy',
    'cross_domain_js_divergence',
    'paired_permutation_test', 'cohens_d',
    # Reasoning Types
    'ReasoningType', 'N_REASONING_TYPES', 'REASONING_TYPES',
    'TYPE_NAMES', 'TYPE_SHORT_NAMES', 'TYPE_COLORS',
    'HeuristicLabeler',
    # Controller
    'GeometricReasoningController', 'ControllerConfig', 'TypeClassifier',
    'TYPE_MERGE_MAP', 'CORE_TYPE_NAMES', 'N_CORE_TYPES', 'merge_labels',
    # ARTI (Active Reasoning Type Identifier)
    'ARTI', 'ARTIConfig', 'StreamingARTI',
    # ARTI v2 (Trajectory-based)
    'ARTIV2', 'ARTIV2Config', 'TrajectoryFeatureExtractor',
    # Text utilities
    'segment_text',
]
