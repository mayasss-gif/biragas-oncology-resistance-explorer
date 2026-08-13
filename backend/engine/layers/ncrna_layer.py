"""
NON-CODING RNA LAYER - how ncRNA becomes CAUSAL, not just measured.

Many pipelines treat lncRNA/miRNA as expression noise. BiRAGAS promotes them to causal hypotheses:
  * a microRNA that represses a target is a SIGNED, DIRECTED edge (miRNA -| target);
  * a lncRNA that regulates in cis/trans is a candidate directed edge;
  * a ceRNA (circRNA/lncRNA) that sponges a miRNA is modeled as a mediator (front-door territory);
  * a variant in a non-coding locus is a Mendelian-randomization instrument, so an ncRNA can be a
    genetically anchored causal DRIVER, not merely a biomarker.

These become orientation PRIORS and NCRNA_REGULATORY evidence records feeding Layers 1 and 4. The
honesty rule: target predictions (TargetScan/miRTarBase, LncBook) are PRIORS, not proof. They seed
orientation and a small LLR; the edge must still pass do-calculus identification and refutation.

Dependency-light: numpy, pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _schema import EvidenceRecord, EvidenceType


def mirna_target_edges(mirna_targets: pd.DataFrame,
                       expr: pd.DataFrame | None = None,
                       context_min_r: float = -0.2) -> list[EvidenceRecord]:
    """
    mirna_targets: rows of (mirna, target, source_db, [score]). Emits directed REPRESSIVE priors
    miRNA -| target. If expression is provided, keep only pairs that are anti-correlated in THIS
    dataset (a miRNA that represses its target should be negatively correlated with it), which turns
    a generic database prior into a context-specific one.
    """
    out = []
    for _, r in mirna_targets.iterrows():
        m, t = r["mirna"], r["target"]
        ctx = None
        if expr is not None and m in expr.columns and t in expr.columns:
            ctx = float(expr[m].corr(expr[t]))
            if ctx > context_min_r:                    # not anti-correlated here -> skip prior
                continue
        out.append(EvidenceRecord(
            EvidenceType.NCRNA_REGULATORY, source=m, target=t,
            statistic=float(r.get("score", 0.5)), direction=-1,
            detail={"kind": "miRNA_repression", "context_corr": ctx,
                    "db": r.get("source_db", "TargetScan/miRTarBase")},
            provenance="miRNA-target prior (context-filtered)"))
    return out


def lncrna_target_edges(lncrna_targets: pd.DataFrame) -> list[EvidenceRecord]:
    """lncrna_targets: (lncrna, target, mode in {cis,trans}, direction in {+1,-1})."""
    return [EvidenceRecord(
        EvidenceType.NCRNA_REGULATORY, source=r["lncrna"], target=r["target"],
        statistic=0.5, direction=int(r.get("direction", 0)),
        detail={"kind": f"lncRNA_{r.get('mode', 'trans')}"}, provenance="lncRNA regulatory prior")
        for _, r in lncrna_targets.iterrows()]


def cerna_mediators(sponge_table: pd.DataFrame) -> list[dict]:
    """
    ceRNA sponging: circRNA/lncRNA sponges a miRNA, relieving repression of the miRNA's target.
    This is a MEDIATED effect (sponge -> miRNA -> target), so it is a front-door structure that
    Layer 4 identification handles. Returns mediator hints for the DAG builder.
    """
    return [{"sponge": r["sponge"], "mirna": r["mirna"], "target": r["target"],
             "structure": "front_door_mediation"} for _, r in sponge_table.iterrows()]


def ncrna_orientation_anchors(records: list[EvidenceRecord], min_score: float = 0.4):
    """[(source, target, method, confidence)] PRIOR anchors (weaker than MR/temporal) for discovery."""
    return [(r.source, r.target, f"ncRNA:{r.detail.get('kind', 'regulatory')}", min(0.9, r.statistic))
            for r in records if r.statistic >= min_score]


def ncrna_as_mr_instrument(ncrna: str, target: str, mr_dat: pd.DataFrame):
    """
    A non-coding locus can be an MR instrument for its target. Hand mr_dat (SNP-level summary stats
    for variants in/near the ncRNA) straight to Layer 3 MendelianRandomization. Returned here as a
    passthrough so the caller wires ncRNA genetics into the strongest causal stream.
    """
    return {"exposure": ncrna, "outcome": target, "instrument_dat": mr_dat,
            "note": "ncRNA treated as a genetically anchored exposure via cis variants"}


if __name__ == "__main__":
    # miR-29 family represses extracellular-matrix genes (COL1A1, FN1) - real anti-fibrotic biology.
    targets = pd.DataFrame({"mirna": ["miR-29a", "miR-29a", "miR-29b"],
                            "target": ["COL1A1", "FN1", "COL3A1"],
                            "source_db": ["miRTarBase"] * 3, "score": [0.8, 0.7, 0.6]})
    rng = np.random.default_rng(0)
    n = 200
    mir = rng.normal(size=n)
    fn1 = -0.7 * mir + rng.normal(0, 0.5, n)       # anti-correlated (repression) in-context
    expr = pd.DataFrame({"miR-29a": mir, "FN1": fn1, "COL1A1": -0.6 * mir + rng.normal(0, .5, n),
                         "COL3A1": rng.normal(size=n)})
    edges = mirna_target_edges(targets, expr)
    print("context-filtered miRNA repressive priors:")
    for e in edges:
        ctx = e.detail.get("context_corr")
        ctx_str = f"{ctx:.2f}" if ctx is not None else "n/a (no in-context expression)"
        print(f"  {e.source} -| {e.target}  ctx_corr={ctx_str}")
    print("orientation anchors:", ncrna_orientation_anchors(edges))
