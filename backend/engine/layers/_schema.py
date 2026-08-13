"""
Shared typed contracts for the BiRAGAS causality-engine EXPANSION modules.

This is a light, standalone mirror of causal_engine/schemas.py so the Codes/ expansion runs on its
own (numpy/pandas/scipy/statsmodels/networkx only). Every layer emits EvidenceRecord objects; the
fusion + CCS layer folds them into a CausalEdge with a calibrated probability and an honest gate.

The point of this file: a causal claim is a TYPED, PROVENANCE-CARRYING object, not a printed string.
Each edge records which stream produced it, the statistic, the identification strategy, the adjusted
effect, the refutation outcomes and the calibrated Causal Confidence Score.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class EvidenceType(str, Enum):
    """Independent evidence streams, ranked by causal strength (weakest = 0 weight)."""
    GENETIC_MR = "genetic_mendelian_randomization"     # nature's randomized trial (strongest)
    CRISPR_INTERVENTIONAL = "crispr_perturbation"       # a real intervention
    COLOCALIZATION = "eqtl_colocalization"              # shared causal variant (PP.H4)
    TEMPORAL_PRECEDENCE = "temporal_causal_discovery"   # cause precedes effect (conditioned)
    CURATED_PRIOR = "signor_curated_directed"           # literature-anchored mechanism
    NCRNA_REGULATORY = "ncrna_regulatory_prior"         # miRNA/lncRNA directed prior
    OBSERVATIONAL_DISCOVERY = "constraint_or_score_discovery"  # PC/FCI/GES/LiNGAM/NOTEARS
    OBSERVATIONAL_ASSOCIATION = "coexpression_association"     # correlation (ZERO weight)


# Bradford-Hill weights used as prior LOG-LIKELIHOOD-RATIO anchors in the Bayesian fusion,
# identical in spirit to causal_engine/schemas.EVIDENCE_PRIOR_LLR. Calibrated against gold later.
EVIDENCE_PRIOR_LLR = {
    EvidenceType.GENETIC_MR: 2.2,
    EvidenceType.CRISPR_INTERVENTIONAL: 2.0,
    EvidenceType.COLOCALIZATION: 1.4,
    EvidenceType.TEMPORAL_PRECEDENCE: 1.0,
    EvidenceType.CURATED_PRIOR: 1.1,
    EvidenceType.NCRNA_REGULATORY: 0.6,       # a prior; must still be identified + refuted
    EvidenceType.OBSERVATIONAL_DISCOVERY: 0.7,
    EvidenceType.OBSERVATIONAL_ASSOCIATION: 0.0,   # correlation contributes nothing, by construction
}


class IdentificationStrategy(str, Enum):
    BACKDOOR = "backdoor_adjustment"
    FRONTDOOR = "frontdoor_criterion"
    IV = "instrumental_variable"
    GENETIC_IV = "mendelian_randomization_iv"
    NOT_IDENTIFIED = "not_identified"


@dataclass
class EvidenceRecord:
    etype: EvidenceType
    source: str                       # gene/exposure
    target: str                       # outcome
    statistic: float = 0.0            # MR beta, coloc PP.H4, edge stability, lag strength, ...
    p_value: Optional[float] = None
    fdr: Optional[float] = None
    direction: int = 0                # +1 up-regulates target, -1 represses, 0 unknown
    detail: dict = field(default_factory=dict)
    provenance: str = ""              # dataset/DB/pipeline that produced it

    def key(self):
        return (self.source, self.target)


@dataclass
class EffectEstimate:
    point: float
    ci_low: float
    ci_high: float
    se: float
    p_value: float
    estimator: str
    n: int = 0


@dataclass
class RefutationResult:
    refuter: str
    passed: bool
    new_effect: Optional[float] = None
    detail: str = ""


@dataclass
class CausalEdge:
    source: str
    target: str
    orientation_method: str = ""
    bootstrap_stability: float = 0.0
    identification: IdentificationStrategy = IdentificationStrategy.NOT_IDENTIFIED
    adjustment_set: list = field(default_factory=list)
    effect: Optional[EffectEstimate] = None
    refutations: list = field(default_factory=list)
    evidence: list = field(default_factory=list)   # list[EvidenceRecord]
    posterior_prob: float = 0.0
    ccs: float = 0.0
    ccs_tier: str = "T3"
    is_causal: bool = False

    def refutations_passed(self) -> bool:
        return bool(self.refutations) and all(r.passed for r in self.refutations)

    def to_row(self) -> dict:
        e = self.effect
        return {
            "source": self.source, "target": self.target,
            "orientation": self.orientation_method,
            "stability": round(self.bootstrap_stability, 3),
            "identification": self.identification.value,
            "adjustment_set": ";".join(self.adjustment_set),
            "effect": None if e is None else round(e.point, 4),
            "refuted_ok": self.refutations_passed(),
            "n_streams": len({ev.etype for ev in self.evidence}),
            "posterior": round(self.posterior_prob, 4),
            "ccs": round(self.ccs, 4), "tier": self.ccs_tier, "is_causal": self.is_causal,
        }
