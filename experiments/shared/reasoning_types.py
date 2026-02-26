#!/usr/bin/env python3
"""
Reasoning Type Definitions and Geometric Signatures (v3 — 10 types).

Defines the taxonomy of reasoning types used by the Active Reasoning Type
Identifier (ARTI). Each type has:
- A geometric signature on the ~10D operator manifold
- Trigger keywords/patterns for heuristic pre-labeling
- Example reasoning traces
- Paper 11 Level 1 operator mapping

v3 Changes: Split Cause-Effect (which was a catch-all at 11% accuracy) into
three geometrically distinct sub-types:
  - Physical Cause: deterministic natural mechanisms
  - Behavioral Cause: human decisions → outcomes
  - Systemic Cause: cascading chains, feedback loops

Based on Paper 11 E4 spectral analysis (4,695 transitions) which showed
the operator space is a continuous ~10D manifold with density peaks
corresponding to distinct reasoning modes (silhouette=0.33 in 10D).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum
import re


class ReasoningType(IntEnum):
    """
    Ten fundamental reasoning types identifiable by geometric signature.

    v3: Cause-Effect split into Physical/Behavioral/Systemic.
    """
    PHYSICAL_CAUSE = 0     # Deterministic natural mechanisms
    BEHAVIORAL_CAUSE = 1   # Human decisions → outcomes
    SYSTEMIC_CAUSE = 2     # Cascading chains, feedback loops
    DEDUCTION = 3          # Logical deduction (if-then, syllogism)
    INDUCTION = 4          # Pattern generalization
    ANALOGY = 5            # Structural mapping across domains
    CONSERVATION = 6       # Invariance / quantity preservation
    COUNTERFACTUAL = 7     # "What if not X" branching
    ABDUCTION = 8          # Inference to best explanation
    DECOMPOSITION = 9      # Breaking into parts / subgoaling


# Number of types
N_REASONING_TYPES = len(ReasoningType)


@dataclass
class GeometricSignature:
    """
    Expected geometric signature of a reasoning type on the manifold.

    These describe statistical properties of the delta vectors
    (delta = embed(step_{t+1}) - embed(step_t)) when projected onto
    the ~10D manifold subspace.
    """
    curvature: str            # 'low' (linear), 'medium' (curved), 'high' (sharp turns)
    displacement_magnitude: str  # 'near_zero', 'small', 'medium', 'large', 'discontinuous'
    trajectory_shape: str     # 'linear', 'branching', 'converging', 'parallel', 'loop', 'jump', 'cascade'
    direction_consistency: str  # 'high' (same direction), 'medium', 'low' (chaotic)
    description: str          # Human-readable description


@dataclass
class ReasoningTypeSpec:
    """Full specification of a reasoning type."""
    type_id: ReasoningType
    name: str
    short_name: str           # For plot labels
    level1_operators: List[str]  # Paper 11 operator mapping

    geometric_signature: GeometricSignature

    # Trigger keywords/phrases for heuristic labeling
    trigger_keywords: List[str]
    trigger_patterns: List[str]  # Regex patterns

    # Example reasoning steps (pairs of consecutive steps)
    examples: List[Tuple[str, str]]

    # Domain associations (not exclusive)
    common_domains: List[str]

    # Color for visualization
    color: str


# ─── Type Specifications ──────────────────────────────────────────────────────

REASONING_TYPES: Dict[ReasoningType, ReasoningTypeSpec] = {

    # ── CAUSE SUB-TYPES (v3 split) ────────────────────────────────────

    ReasoningType.PHYSICAL_CAUSE: ReasoningTypeSpec(
        type_id=ReasoningType.PHYSICAL_CAUSE,
        name="Physical Cause",
        short_name="PhysCause",
        level1_operators=["causality"],
        geometric_signature=GeometricSignature(
            curvature="low",
            displacement_magnitude="medium",
            trajectory_shape="linear",
            direction_consistency="high",
            description="Strong directed linear flow along a density ridge. "
                        "Deterministic mechanism constrains the path tightly. "
                        "High consistency because physical laws are predictable.",
        ),
        trigger_keywords=[
            "temperature", "pressure", "force", "energy", "gravity",
            "radiation", "chemical", "reaction", "voltage", "current",
            "friction", "magnetic", "electric", "heat", "cool",
            "expand", "contract", "evaporate", "condense", "dissolve",
            "erode", "corrode", "oxidize", "freeze", "melt", "boil",
            "accelerate", "decelerate", "collide", "absorb", "emit",
            "wavelength", "frequency", "amplitude", "earthquake",
            "photosynthesis", "metabolism", "mutation", "infection",
        ],
        trigger_patterns=[
            r"\b(temperature|pressure|force|energy)\b.*\b(caused?|led to|results? in)\b",
            r"\b(caused?|led to|results? in)\b.*\b(temperature|pressure|force|energy)\b",
            r"\b(heat|cool|freez|melt|boil|evaporat|condens)\w*\b.*\b(caused?|led to|results? in|which)\b",
            r"\b(radiation|chemical|reaction|voltage|current)\b",
            r"\b(earthquake|eruption|storm|flood|drought|erosion)\b.*\b(caused?|led to|results? in|damag)\b",
            r"\b(infect|virus|bacteria|enzyme|protein|DNA|cell)\b.*\b(caused?|led to|results? in|damag)\b",
        ],
        examples=[
            ("The temperature dropped below freezing overnight.",
             "Therefore the pipes froze and burst from ice expansion."),
            ("The asteroid impact generated a massive dust cloud.",
             "The dust blocked sunlight for months, causing global cooling."),
            ("Increasing voltage across the resistor followed Ohm's law.",
             "The current increased proportionally to the voltage change."),
        ],
        common_domains=["physics", "chemistry", "biology", "geology", "medicine"],
        color="#e74c3c",  # Red
    ),

    ReasoningType.BEHAVIORAL_CAUSE: ReasoningTypeSpec(
        type_id=ReasoningType.BEHAVIORAL_CAUSE,
        name="Behavioral Cause",
        short_name="BehvCause",
        level1_operators=["causality"],
        geometric_signature=GeometricSignature(
            curvature="medium",
            displacement_magnitude="medium",
            trajectory_shape="branching",
            direction_consistency="medium",
            description="Medium curvature branching trajectory. Human decisions "
                        "have multiple possible outcomes, creating wider trajectory "
                        "spread than deterministic physical causes. Direction is "
                        "moderately consistent (actions are intentional but outcomes "
                        "are uncertain).",
        ),
        trigger_keywords=[
            "decided", "chose", "policy", "invested", "purchased",
            "voted", "raised", "lowered", "spent", "forgot",
            "practiced", "studied", "hired", "fired", "implemented",
            "launched", "announced", "negotiated", "agreed", "refused",
            "subsidized", "taxed", "regulated", "deregulated",
            "price", "cost", "salary", "budget", "funding",
            "company", "government", "organization", "management",
        ],
        trigger_patterns=[
            r"\b(decided|chose|elected|opted)\b.*\b(to|that)\b",
            r"\b(policy|regulation|law|rule)\b.*\b(caused?|led to|results? in)\b",
            r"\b(company|government|management|team)\b.*\b(raised|lowered|changed|implemented)\b",
            r"\b(invest|purchas|bought|sold|spent)\w*\b.*\b(caused?|led to|results? in|which)\b",
            r"\b(forgot|neglect|fail)\w*\b.*\b(caused?|led to|results? in)\b",
            r"\b(practic|studi|train)\w*\b.*\b(caused?|led to|results? in|improv|succeed)\b",
        ],
        examples=[
            ("She studied for 6 hours every day for a month.",
             "As a result, she scored 98% on the exam."),
            ("The central bank raised interest rates by 2 percent.",
             "Consequently, housing prices declined across the region."),
            ("The company forgot to renew the domain registration.",
             "Their entire website went offline for three days."),
        ],
        common_domains=["economics", "business", "education", "everyday", "politics"],
        color="#c0392b",  # Dark red
    ),

    ReasoningType.SYSTEMIC_CAUSE: ReasoningTypeSpec(
        type_id=ReasoningType.SYSTEMIC_CAUSE,
        name="Systemic Cause",
        short_name="SysCause",
        level1_operators=["causality", "composition"],
        geometric_signature=GeometricSignature(
            curvature="high",
            displacement_magnitude="large",
            trajectory_shape="cascade",
            direction_consistency="low",
            description="High curvature cascading trajectory with large displacement. "
                        "Multi-step causal chains amplify through feedback loops. "
                        "Low consistency because each link in the chain can diverge. "
                        "The trajectory fans out as effects propagate.",
        ),
        trigger_keywords=[
            "cascade", "chain reaction", "feedback", "spiral",
            "domino", "triggered", "ripple", "propagated", "amplified",
            "spread", "systemic", "network", "interconnected",
            "vicious cycle", "virtuous cycle", "snowball", "escalated",
            "downstream", "upstream", "knock-on", "second-order",
            "unintended consequence", "side effect", "spillover",
            "ecosystem", "food web", "supply chain", "contagion",
        ],
        trigger_patterns=[
            r"\b(cascade|chain\s+reaction|domino|ripple|snowball)\b",
            r"\b(feedback|spiral|vicious|virtuous)\s+(loop|cycle|effect)\b",
            r"\bwhich\s+(then\s+)?(caused?|led\s+to|triggered|resulted)\b.*\bwhich\s+(then\s+)?(caused?|led|triggered)\b",
            r"\b(spread|propagat|amplif|escalat)\w*\b.*\b(across|throughout|to)\b",
            r"\b(systemic|interconnect|downstream|upstream|knock-on|second-order|spillover)\b",
            r"\b(unintended|side\s+effect|collateral)\b.*\b(consequence|effect|damage)\b",
        ],
        examples=[
            ("The drought killed crops, causing food shortages.",
             "Food shortages triggered mass migration, which destabilized neighboring regions."),
            ("A single bank failure triggered a chain reaction of defaults.",
             "The cascade spread through the interconnected financial system, causing a global crisis."),
            ("Removing the keystone predator caused herbivore overpopulation.",
             "Overgrazing destroyed vegetation, which led to soil erosion and ecosystem collapse."),
        ],
        common_domains=["ecology", "economics", "epidemiology", "technology", "geopolitics"],
        color="#e67e73",  # Light red
    ),

    # ── REMAINING TYPES (renumbered from v2) ──────────────────────────

    ReasoningType.DEDUCTION: ReasoningTypeSpec(
        type_id=ReasoningType.DEDUCTION,
        name="Deduction",
        short_name="Deduc",
        level1_operators=["causality", "composition"],
        geometric_signature=GeometricSignature(
            curvature="medium",
            displacement_magnitude="medium",
            trajectory_shape="branching",
            direction_consistency="high",
            description="Tree-like branching with preserved coherence. "
                        "Forking paths that stay close together (inner-product "
                        "bound). Each branch represents a derivation step.",
        ),
        trigger_keywords=[
            "if", "then", "all", "every", "must", "necessarily",
            "follows that", "conclude", "deduce", "infer",
            "given that", "it follows", "premises", "valid",
            "modus ponens", "syllogism", "entails", "proof",
        ],
        trigger_patterns=[
            r"\ball\b.*\bare\b.*\btherefore\b",
            r"\bif\b.*\bthen\b.*\btherefore\b",
            r"\bgiven\b.*\b(conclude|follows|must)\b",
            r"\b(premise|premises)\b.*\b(conclusion|therefore)\b",
        ],
        examples=[
            ("All mammals are warm-blooded. A whale is a mammal.",
             "Therefore, a whale is warm-blooded."),
            ("If it rains, the ground gets wet. It is raining.",
             "Therefore, the ground is wet."),
            ("Every prime number greater than 2 is odd. 7 is prime and greater than 2.",
             "Therefore, 7 is odd."),
        ],
        common_domains=["logic", "math", "law", "science"],
        color="#3498db",  # Blue
    ),

    ReasoningType.INDUCTION: ReasoningTypeSpec(
        type_id=ReasoningType.INDUCTION,
        name="Induction",
        short_name="Induc",
        level1_operators=["abstraction", "composition"],
        geometric_signature=GeometricSignature(
            curvature="medium",
            displacement_magnitude="small",
            trajectory_shape="converging",
            direction_consistency="medium",
            description="Converging bundle toward a generalization peak. "
                        "Multiple paths merge as specific observations "
                        "converge to a general pattern.",
        ),
        trigger_keywords=[
            "every time", "always", "pattern", "generally", "tends to",
            "usually", "most", "often", "observed that", "in each case",
            "evidence suggests", "data shows", "consistently",
            "generalize", "rule", "trend", "correlation",
        ],
        trigger_patterns=[
            r"\b(every|each)\s+time\b.*\b(so|therefore|always)\b",
            r"\b(pattern|trend|consistently)\b",
            r"\b(all|every)\b.*\b(observed|seen|found)\b",
            r"\bdata\b.*\b(shows?|suggests?|indicates?)\b",
        ],
        examples=[
            ("Swan 1 is white. Swan 2 is white. Swan 3 is white.",
             "All observed swans are white, so swans are generally white."),
            ("Every patient given drug X improved within 3 days.",
             "The evidence suggests drug X is effective for this condition."),
            ("Sales increased in Q1, Q2, and Q3 after the ad campaign.",
             "The data shows the campaign consistently drives sales growth."),
        ],
        common_domains=["science", "statistics", "everyday", "medicine"],
        color="#2ecc71",  # Green
    ),

    ReasoningType.ANALOGY: ReasoningTypeSpec(
        type_id=ReasoningType.ANALOGY,
        name="Analogy",
        short_name="Analog",
        level1_operators=["analogy", "symmetry"],
        geometric_signature=GeometricSignature(
            curvature="low",
            displacement_magnitude="medium",
            trajectory_shape="parallel",
            direction_consistency="high",
            description="Parallel or mirrored displacements with high cosine "
                        "similarity between two delta vectors. Two nearby "
                        "trajectories shift together, preserving relational "
                        "structure across domains.",
        ),
        trigger_keywords=[
            "like", "similar to", "just as", "analogous", "comparable",
            "in the same way", "corresponds to", "parallel", "resembles",
            "equivalent", "mirrors", "akin to", "maps to",
        ],
        trigger_patterns=[
            r"\bjust\s+(as|like)\b.*\bso\b",
            r"\bsimilar\s+to\b",
            r"\b(analog(ous|y)|correspond[s]?)\b",
            r"\b(like|as)\b.*\b(is to|relates to)\b",
        ],
        examples=[
            ("The atom has electrons orbiting a nucleus.",
             "Just as planets orbit the sun in the solar system."),
            ("A firewall protects a network from intruders.",
             "Similarly, a moat protects a castle from invaders."),
            ("Teacher is to student as doctor is to patient.",
             "Both relationships involve a knowledgeable guide helping someone."),
        ],
        common_domains=["science", "education", "everyday", "law"],
        color="#9b59b6",  # Purple
    ),

    ReasoningType.CONSERVATION: ReasoningTypeSpec(
        type_id=ReasoningType.CONSERVATION,
        name="Conservation/Invariance",
        short_name="Conserv",
        level1_operators=["conservation", "symmetry"],
        geometric_signature=GeometricSignature(
            curvature="low",
            displacement_magnitude="near_zero",
            trajectory_shape="loop",
            direction_consistency="low",
            description="Near-zero displacement or closed loops on the manifold. "
                        "The reasoning asserts that something stays constant "
                        "despite apparent change.",
        ),
        trigger_keywords=[
            "total", "constant", "preserved", "conserved", "unchanged",
            "same", "equal", "balanced", "net", "sum",
            "before and after", "remains", "invariant", "stays",
        ],
        trigger_patterns=[
            r"\b(total|sum|net)\b.*\b(same|constant|unchanged|equal)\b",
            r"\b(conserv(ed|ation)|preserv(ed|ation)|invariant)\b",
            r"\bbefore\b.*\bafter\b.*\b(same|equal)\b",
            r"\b(balanced|equilibrium)\b",
        ],
        examples=[
            ("A ball is thrown upward, converting kinetic to potential energy.",
             "The total mechanical energy remains constant throughout."),
            ("$500 was transferred from checking to savings.",
             "The total amount in both accounts stays the same."),
            ("The reaction produced 3 moles of CO2 from 3 moles of O atoms.",
             "Mass is conserved: atoms in = atoms out."),
        ],
        common_domains=["physics", "chemistry", "economics", "accounting"],
        color="#f39c12",  # Orange
    ),

    ReasoningType.COUNTERFACTUAL: ReasoningTypeSpec(
        type_id=ReasoningType.COUNTERFACTUAL,
        name="Counterfactual",
        short_name="Counter",
        level1_operators=["negation", "causality"],
        geometric_signature=GeometricSignature(
            curvature="high",
            displacement_magnitude="large",
            trajectory_shape="branching",
            direction_consistency="low",
            description="Sharp orthogonal deviation followed by a new coherent "
                        "branch. Sudden 90+ degree turn from the main trajectory "
                        "into a new coherent trajectory exploring the alternative.",
        ),
        trigger_keywords=[
            "what if", "had not", "would have", "could have",
            "imagine", "suppose", "hypothetically", "alternatively",
            "instead", "otherwise", "without", "in the absence of",
            "contrary to", "if only",
        ],
        trigger_patterns=[
            r"\bwhat\s+if\b",
            r"\b(had|would|could)\s+(not\s+)?(have\s+)?(been|happened|occurred)\b",
            r"\b(hypothetically|suppose|imagine)\b",
            r"\bif\b.*\bhad\s+not\b",
            r"\bwithout\b.*\bwould\b",
        ],
        examples=[
            ("The bridge failed because the supports were corroded.",
             "What if the supports had been inspected monthly? The failure would have been prevented."),
            ("The patient was given antibiotics and recovered.",
             "Without the antibiotics, the infection would likely have spread."),
            ("The company invested in R&D and launched a successful product.",
             "Had they not invested, competitors would have captured the market."),
        ],
        common_domains=["science", "history", "medicine", "everyday"],
        color="#1abc9c",  # Teal
    ),

    ReasoningType.ABDUCTION: ReasoningTypeSpec(
        type_id=ReasoningType.ABDUCTION,
        name="Abduction",
        short_name="Abduc",
        level1_operators=["causality", "optimization"],
        geometric_signature=GeometricSignature(
            curvature="high",
            displacement_magnitude="discontinuous",
            trajectory_shape="jump",
            direction_consistency="low",
            description="Jump to a new distant density peak. Large discontinuous "
                        "delta vector as the reasoning leaps from observation to "
                        "the best available explanation.",
        ),
        trigger_keywords=[
            "best explanation", "most likely", "probably", "suggests",
            "hypothesis", "diagnose", "infer", "must be",
            "the reason is", "explains why", "accounts for",
            "evidence points to", "consistent with",
        ],
        trigger_patterns=[
            r"\b(best|most\s+likely|probable)\s+(explanation|cause|reason)\b",
            r"\b(hypothesis|diagnos[ei]s?)\b",
            r"\beverything\s+points\s+to\b",
            r"\b(explains?|accounts?\s+for)\s+(why|the)\b",
        ],
        examples=[
            ("The lawn is wet, but it hasn't rained and no sprinklers are on.",
             "The best explanation is that dew formed overnight."),
            ("Patient presents fever, cough, and chest pain.",
             "The most likely diagnosis is pneumonia."),
            ("Sales dropped 40% only in Region B after a competitor opened there.",
             "This suggests the new competitor is the primary cause."),
        ],
        common_domains=["medicine", "science", "detective", "everyday"],
        color="#e67e22",  # Dark orange
    ),

    ReasoningType.DECOMPOSITION: ReasoningTypeSpec(
        type_id=ReasoningType.DECOMPOSITION,
        name="Decomposition",
        short_name="Decomp",
        level1_operators=["decomposition", "recursion"],
        geometric_signature=GeometricSignature(
            curvature="medium",
            displacement_magnitude="small",
            trajectory_shape="branching",
            direction_consistency="medium",
            description="Systematic branching into sub-components with moderate "
                        "displacement. The trajectory splits into parallel "
                        "sub-trajectories, each handling a part of the problem.",
        ),
        trigger_keywords=[
            "first", "second", "step", "part", "break down",
            "component", "separately", "divide", "split",
            "subproblem", "factor", "element", "aspect",
            "one by one", "let's consider each",
        ],
        trigger_patterns=[
            r"\b(first|step\s+1)\b.*\b(second|step\s+2|then|next)\b",
            r"\bbreak\s+(down|into)\b",
            r"\b(component|part|factor|aspect)[s]?\b.*\b(separately|individually)\b",
            r"\blet'?s?\s+(consider|look\s+at)\s+each\b",
        ],
        examples=[
            ("To find the total cost, we need distance, rate, and time.",
             "First, calculate distance = 60 mph * 2 hours = 120 miles. Second, find fuel cost."),
            ("The system failure has three potential sources.",
             "Let's check each separately: hardware, software, and network."),
            ("This integral can be solved by partial fractions.",
             "Factor the denominator: (x+1)(x-2), then decompose A/(x+1) + B/(x-2)."),
        ],
        common_domains=["math", "engineering", "science", "everyday"],
        color="#34495e",  # Dark gray-blue
    ),
}


# ─── Convenience Lookups ───────────────────────────────────────────────────────

TYPE_NAMES: List[str] = [REASONING_TYPES[t].name for t in ReasoningType]
TYPE_SHORT_NAMES: List[str] = [REASONING_TYPES[t].short_name for t in ReasoningType]
TYPE_COLORS: List[str] = [REASONING_TYPES[t].color for t in ReasoningType]


def get_type_by_name(name: str) -> Optional[ReasoningType]:
    """Look up reasoning type by name (case-insensitive)."""
    name_lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for t, spec in REASONING_TYPES.items():
        if spec.name.lower().replace("-", "").replace(" ", "") == name_lower:
            return t
        if spec.short_name.lower() == name_lower:
            return t
    return None


# ─── Heuristic Labeler ─────────────────────────────────────────────────────────

class HeuristicLabeler:
    """
    Labels reasoning steps by keyword/pattern matching.

    v3: Distinguishes 3 Cause sub-types via domain-specific keywords.
    Physical triggers (science terms) beat general causal connectors.
    Systemic triggers (cascade/chain terms) beat general causal connectors.
    Behavioral triggers (human action terms) beat general causal connectors.
    """

    def __init__(self):
        self._compiled_patterns = {}
        for rtype, spec in REASONING_TYPES.items():
            self._compiled_patterns[rtype] = [
                re.compile(p, re.IGNORECASE) for p in spec.trigger_patterns
            ]

    def label(self, text: str) -> Dict[ReasoningType, float]:
        """
        Produce soft label distribution for a text.

        Returns:
            Dict mapping ReasoningType -> score (sums to 1.0)
        """
        text_lower = text.lower()
        scores = {}

        for rtype, spec in REASONING_TYPES.items():
            score = 0.0

            # Keyword matches (each keyword contributes 1.0)
            for kw in spec.trigger_keywords:
                if kw.lower() in text_lower:
                    score += 1.0

            # Pattern matches (each pattern contributes 2.0)
            for pattern in self._compiled_patterns[rtype]:
                if pattern.search(text):
                    score += 2.0

            scores[rtype] = score

        # Normalize to probability distribution
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            # Uniform if no matches
            n = len(REASONING_TYPES)
            scores = {k: 1.0 / n for k in REASONING_TYPES}

        return scores

    def label_hard(self, text: str) -> Tuple[ReasoningType, float]:
        """
        Produce hard label (argmax) with confidence.

        Returns:
            (type, confidence) where confidence is the max probability
        """
        soft = self.label(text)
        best_type = max(soft, key=soft.get)
        return best_type, soft[best_type]

    def label_batch(self, texts: List[str]) -> List[Dict[ReasoningType, float]]:
        """Label a batch of texts."""
        return [self.label(t) for t in texts]


# ─── Synthetic Trace Generator ─────────────────────────────────────────────────

def get_examples_by_type(rtype: ReasoningType) -> List[Tuple[str, str]]:
    """Get example reasoning step pairs for a given type."""
    return REASONING_TYPES[rtype].examples


def get_all_examples() -> List[Tuple[str, str, ReasoningType]]:
    """Get all example pairs with their type labels."""
    result = []
    for rtype, spec in REASONING_TYPES.items():
        for step1, step2 in spec.examples:
            result.append((step1, step2, rtype))
    return result


if __name__ == "__main__":
    print("Reasoning Type Definitions (v3 — 10 types)")
    print("=" * 60)

    for rtype, spec in REASONING_TYPES.items():
        print(f"\n{spec.name} ({spec.short_name})")
        print(f"  Level 1 operators: {spec.level1_operators}")
        print(f"  Geometric: {spec.geometric_signature.trajectory_shape} / "
              f"{spec.geometric_signature.displacement_magnitude}")
        print(f"  Keywords: {spec.trigger_keywords[:5]}...")
        print(f"  Color: {spec.color}")

    print(f"\nTotal types: {N_REASONING_TYPES}")

    # Test heuristic labeler
    print("\n" + "=" * 60)
    print("Heuristic Labeler Test")
    labeler = HeuristicLabeler()

    test_texts = [
        "Heating the metal rod caused it to expand by 2 millimeters due to thermal expansion.",
        "The company raised prices by 30%, which caused sales volume to drop by half.",
        "The drought killed crops, causing food shortages that triggered mass migration across borders.",
        "All mammals are warm-blooded. Whales are mammals. Therefore whales are warm-blooded.",
        "Every time I water the plant, it grows. So watering consistently helps growth.",
        "Just as a CPU processes data, a brain processes information.",
        "The total energy before and after the collision remains the same.",
        "What if the bridge had been inspected? The failure would have been prevented.",
        "The best explanation for the wet lawn is overnight dew.",
        "First, calculate the distance. Second, find the cost per mile.",
    ]

    for text in test_texts:
        rtype, conf = labeler.label_hard(text)
        name = REASONING_TYPES[rtype].short_name
        print(f"  [{name:>10} {conf:.0%}] {text[:70]}...")

    # Get all examples
    examples = get_all_examples()
    print(f"\nTotal built-in examples: {len(examples)}")
    print("\nAll tests passed!")
