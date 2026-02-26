#!/usr/bin/env python3
"""
Test ARTI v1 vs v2 on 50 new reasoning phrases (5 per type, never seen in training).

All phrases are freshly written — not from EXPANDED_EXAMPLES or any training source.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from collections import Counter

from shared.reasoning_types import (
    ReasoningType, N_REASONING_TYPES, REASONING_TYPES,
    TYPE_SHORT_NAMES,
)
from shared.arti import ARTI, ARTIConfig
from shared.arti_v2 import ARTIV2, ARTIV2Config
from shared.encoder import SentenceTransformerEncoder
from shared.text_utils import segment_text

BASE_DIR = Path(__file__).parent
V1_MODEL = BASE_DIR / "results" / "arti_v3" / "arti_model.pt"
V2_MODEL = BASE_DIR / "results" / "arti_v2_trajectory" / "arti_v2_model.pt"

# ─── 50 New Test Phrases (5 per type, never in training data) ────────────────

TEST_PHRASES = [
    # ── PhysCause (0) ────────────────────────────────────────────────
    ("Rapid cooling of the molten glass created internal stress fractures throughout the pane.",
     "PhysCause"),
    ("The intense gamma radiation ionized the surrounding air molecules within milliseconds.",
     "PhysCause"),
    ("Prolonged UV exposure degraded the polymer chains, making the plastic brittle and cracked.",
     "PhysCause"),
    ("The supersonic shockwave shattered every window within a two-kilometer radius of the blast.",
     "PhysCause"),
    ("Capillary action drew the water upward through the narrow tubes against the pull of gravity.",
     "PhysCause"),

    # ── BehvCause (1) ────────────────────────────────────────────────
    ("The mayor's decision to close the downtown bridge diverted traffic and doubled commute times.",
     "BehvCause"),
    ("After the coach switched to a zone defense, the opposing team's scoring dropped by 30 percent.",
     "BehvCause"),
    ("The CEO chose to delay the product launch by three months, which allowed competitors to gain ground.",
     "BehvCause"),
    ("Parents who read aloud to their children every night produced kids with larger vocabularies by age five.",
     "BehvCause"),
    ("The committee voted to abolish the entrance exam, resulting in a 40 percent surge in applications.",
     "BehvCause"),

    # ── SysCause (2) ────────────────────────────────────────────────
    ("A single contaminated ingredient entered the supply chain, sickened hundreds, triggered a nationwide recall, and bankrupted the distributor.",
     "SysCause"),
    ("Deforestation upstream increased sediment runoff, which clogged the dam's intake, which reduced hydroelectric output, which caused rolling blackouts downstream.",
     "SysCause"),
    ("The initial layoffs reduced consumer spending, which lowered retail revenue, which forced more layoffs in a self-reinforcing downward spiral.",
     "SysCause"),
    ("One false rumor on social media triggered panic buying, which emptied shelves, which fueled more panic in an escalating feedback loop.",
     "SysCause"),
    ("The bee population collapse reduced pollination, which shrank fruit harvests, which raised food prices, which destabilized rural economies across the region.",
     "SysCause"),

    # ── Deduction (3) ────────────────────────────────────────────────
    ("All elements in group 1 of the periodic table are alkali metals. Sodium is in group 1. Therefore, sodium is an alkali metal.",
     "Deduc"),
    ("If the circuit is open, no current flows. The multimeter reads zero amperes. Therefore, the circuit must be open.",
     "Deduc"),
    ("Every employee who passes the certification earns a bonus. Raj passed the certification. Therefore, Raj earns a bonus.",
     "Deduc"),
    ("Parallel lines never intersect. These two rails are parallel. Therefore, they will never meet.",
     "Deduc"),
    ("Any triangle inscribed in a semicircle has a right angle opposite the diameter. This triangle is inscribed in a semicircle. Therefore, it contains a right angle.",
     "Deduc"),

    # ── Induction (4) ────────────────────────────────────────────────
    ("In every clinical trial across four continents, the vaccine reduced hospitalization by at least 85 percent. The pattern strongly supports its global efficacy.",
     "Induc"),
    ("Each of the fifteen bridges inspected this year showed fatigue cracks near the welds. The consistent finding suggests a systemic manufacturing defect.",
     "Induc"),
    ("Every startup in our portfolio that hired a CFO before Series B survived past year five. This pattern indicates early financial leadership matters.",
     "Induc"),
    ("Soil samples from 30 different farms all showed elevated phosphorus after switching to the new fertilizer. The data consistently confirms phosphorus enrichment.",
     "Induc"),
    ("We measured the alloy's tensile strength 200 times and got values within 2 percent of 450 MPa each time. The evidence points to a reliable material property.",
     "Induc"),

    # ── Analogy (5) ──────────────────────────────────────────────────
    ("A thermostat regulates room temperature in the same way a governor regulates an engine's speed — both use negative feedback loops.",
     "Analog"),
    ("Training a neural network is analogous to sculpting marble: you start with excess material and gradually remove what doesn't belong.",
     "Analog"),
    ("The relationship between a compiler and source code mirrors the relationship between a translator and a foreign text.",
     "Analog"),
    ("Just as tectonic plates float on the mantle and occasionally collide, market sectors drift and periodically disrupt each other.",
     "Analog"),
    ("Antibodies binding to a virus are comparable to a lock and key mechanism where only the right shape fits.",
     "Analog"),

    # ── Conservation (6) ─────────────────────────────────────────────
    ("When the skater spins faster by pulling her arms in, angular momentum stays constant because the moment of inertia decreases.",
     "Conserv"),
    ("Transferring 200 gigabytes from the main server to the backup changes the storage distribution but not the total data volume.",
     "Conserv"),
    ("In every nuclear decay we observed, the total baryon number before and after remained exactly the same.",
     "Conserv"),
    ("Redistributing budget from marketing to engineering changed departmental allocations but the company's total spending stayed at 50 million.",
     "Conserv"),
    ("The chemical reaction rearranged atoms into new molecules, yet the total mass on the scale didn't change by even a microgram.",
     "Conserv"),

    # ── Counterfactual (7) ───────────────────────────────────────────
    ("If the Titanic had carried enough lifeboats, the death toll would have been a fraction of the actual 1,500 casualties.",
     "Counter"),
    ("Had the programmer included input validation, the SQL injection attack would never have succeeded.",
     "Counter"),
    ("What if antibiotics had never been discovered? Simple infections that we cure today would still be lethal.",
     "Counter"),
    ("Without the invention of the transistor in 1947, modern computing as we know it could not exist.",
     "Counter"),
    ("If the reactor's coolant pump had not failed, the meltdown at the plant would have been entirely avoidable.",
     "Counter"),

    # ── Abduction (8) ────────────────────────────────────────────────
    ("The telescope detected unusual dimming patterns around the distant star. The most plausible explanation is a transiting exoplanet.",
     "Abduc"),
    ("Network latency spikes only between 2 and 4 AM when batch jobs run. The likely culprit is resource contention from the nightly ETL pipeline.",
     "Abduc"),
    ("Several unrelated witnesses reported identical symptoms after visiting the same restaurant. Food contamination is the best explanation.",
     "Abduc"),
    ("The ancient pottery fragments found on both sides of the strait suggest these separated civilizations once traded across the water.",
     "Abduc"),
    ("The patient improves every time the medication is administered and relapses when it's withdrawn. This pattern most likely confirms the drug's therapeutic effect.",
     "Abduc"),

    # ── Decomposition (9) ────────────────────────────────────────────
    ("To migrate the legacy system, first inventory all dependencies, then containerize each service, then set up orchestration, and finally run parallel testing.",
     "Decomp"),
    ("Diagnosing the engine failure requires checking ignition, fuel delivery, and compression independently before drawing a conclusion.",
     "Decomp"),
    ("The research proposal breaks into four sections: literature review, methodology, expected results, and timeline with milestones.",
     "Decomp"),
    ("To calculate the spacecraft's trajectory, separately compute gravitational influences from Earth, Moon, and Sun, then sum the vectors.",
     "Decomp"),
    ("Solving the supply chain bottleneck means analyzing procurement, manufacturing, warehousing, and logistics as four independent workstreams.",
     "Decomp"),
]


def main():
    print("=" * 80)
    print("ARTI v1 vs v2 — 50 New Reasoning Phrases Test")
    print("=" * 80)

    # Load encoder
    print("\nLoading encoder...")
    encoder = SentenceTransformerEncoder(
        model_name="all-MiniLM-L6-v2", hidden_dim=256, load_pretrained=True,
    )

    # Load v1
    print("Loading ARTI v1...")
    v1_config = ARTIConfig(
        encoder_dim=384, manifold_dim=10,
        n_geometric_features=32, classifier_type='mlp',
        hidden_dim=64, dropout=0.1,
    )
    v1 = ARTI(v1_config)
    v1.load_state_dict(torch.load(str(V1_MODEL), weights_only=True, map_location='cpu'))
    v1.eval()

    # Load v2
    print("Loading ARTI v2...")
    v2_config = ARTIV2Config(
        encoder_dim=384, manifold_dim=10,
        traj_feature_dim=60, traj_hidden=48,
        n_classes=N_REASONING_TYPES, classifier_hidden=96, dropout=0.1,
    )
    v2 = ARTIV2(v2_config)
    v2.load_state_dict(torch.load(str(V2_MODEL), weights_only=True, map_location='cpu'))
    v2.eval()

    # Run all tests
    print(f"\nTesting {len(TEST_PHRASES)} phrases...\n")
    print(f"{'#':>3}  {'Expected':>10}  {'v1_pred':>10} {'v1%':>5}  {'v2_pred':>10} {'v2%':>5}  {'v1':>3} {'v2':>3}  Text")
    print("-" * 130)

    v1_correct = 0
    v2_correct = 0
    v1_per_type = Counter()
    v2_per_type = Counter()
    v1_per_type_total = Counter()
    v2_per_type_total = Counter()
    disagreements = []

    for i, (text, expected) in enumerate(TEST_PHRASES):
        # v1 classification
        with torch.no_grad():
            emb = encoder.encode_texts([text])
            v1_result = v1(emb)
        v1_idx = int(v1_result['type'].item())
        v1_name = REASONING_TYPES[ReasoningType(v1_idx)].short_name
        v1_conf = float(v1_result['confidence'].item())

        # v2 classification
        segments = segment_text(text)
        with torch.no_grad():
            clause_embs = encoder.encode_texts(segments)
            v2_result = v2([clause_embs])
        v2_idx = int(v2_result['type'].item())
        v2_name = REASONING_TYPES[ReasoningType(v2_idx)].short_name
        v2_conf = float(v2_result['confidence'].item())

        v1_ok = v1_name == expected
        v2_ok = v2_name == expected
        if v1_ok: v1_correct += 1
        if v2_ok: v2_correct += 1

        v1_per_type_total[expected] += 1
        v2_per_type_total[expected] += 1
        if v1_ok: v1_per_type[expected] += 1
        if v2_ok: v2_per_type[expected] += 1

        v1_mark = "OK" if v1_ok else "X"
        v2_mark = "OK" if v2_ok else "X"

        short_text = text[:55] + "..." if len(text) > 55 else text
        print(f"{i+1:3}  {expected:>10}  {v1_name:>10} {v1_conf:4.0%}  {v2_name:>10} {v2_conf:4.0%}  {v1_mark:>3} {v2_mark:>3}  {short_text}")

        if v1_idx != v2_idx:
            disagreements.append((i+1, text[:60], expected, v1_name, v2_name))

    # ─── Summary ──────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"  v1 accuracy: {v1_correct}/{len(TEST_PHRASES)} = {v1_correct/len(TEST_PHRASES):.1%}")
    print(f"  v2 accuracy: {v2_correct}/{len(TEST_PHRASES)} = {v2_correct/len(TEST_PHRASES):.1%}")
    delta = v2_correct - v1_correct
    print(f"  delta:       {delta:+d} ({delta/len(TEST_PHRASES):+.1%})")

    print(f"\n{'Type':>10}  {'v1':>8}  {'v2':>8}  {'delta':>7}  {'winner':>6}")
    print(f"  {'-'*48}")
    v2_wins = 0
    v1_wins = 0
    for name in TYPE_SHORT_NAMES:
        total = v1_per_type_total.get(name, 0)
        if total == 0:
            continue
        v1a = v1_per_type.get(name, 0) / total
        v2a = v2_per_type.get(name, 0) / total
        d = v2a - v1a
        winner = "v2" if d > 0.01 else ("v1" if d < -0.01 else "tie")
        if d > 0.01: v2_wins += 1
        elif d < -0.01: v1_wins += 1
        print(f"  {name:>10}  {v1a:7.0%}  {v2a:7.0%}  {d:+6.0%}  {winner:>6}")

    print(f"\n  v2 wins: {v2_wins} types  |  v1 wins: {v1_wins} types  |  ties: {N_REASONING_TYPES - v2_wins - v1_wins}")

    if disagreements:
        print(f"\n{'='*80}")
        print(f"DISAGREEMENTS ({len(disagreements)} / {len(TEST_PHRASES)})")
        print(f"{'='*80}")
        for num, text, expected, v1n, v2n in disagreements:
            correct_model = "v2" if v2n == expected else ("v1" if v1n == expected else "neither")
            print(f"  #{num:2d} Expected={expected:>10}  v1={v1n:>10}  v2={v2n:>10}  correct={correct_model}")
            print(f"      {text}")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
