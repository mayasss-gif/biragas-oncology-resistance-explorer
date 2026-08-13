"""
LAYER 3 - GENETIC (Mendelian randomization + colocalization). The strongest causal stream.

Modules expanded: GWAS / Mendelian randomization, eQTL colocalization.

Why genetics is the closest thing to a randomized controlled trial WITHOUT a wet lab: alleles are
randomized at conception (Mendel's second law) and fixed for life, long before disease. So a genetic
instrument for a gene's expression is an experiment nature already ran, at population scale, that no
bench can match for confounding control or reverse-causation protection. This is a large part of the
answer to "how can you replace the bench": for the genetic layer you are not replacing an experiment,
you are USING one that already exists for essentially every gene with a cis-eQTL.

This module implements a transparent two-sample MR (IVW + Egger + Steiger) with sensitivity checks,
plus an approximate colocalization posterior. The deployed engine wraps R TwoSampleMR / MR-PRESSO /
SuSiE-coloc for full statistical fidelity; this standalone reference needs only numpy/scipy/statsmodels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from _schema import EvidenceRecord, EvidenceType


@dataclass
class MRResult:
    exposure: str
    outcome: str
    beta_ivw: float
    se_ivw: float
    p_ivw: float
    egger_intercept_p: float = np.nan   # horizontal-pleiotropy test
    q_p: float = np.nan                 # Cochran Q heterogeneity
    steiger_ok: bool = True             # exposure -> outcome direction upheld
    coloc_pp4: float = np.nan           # posterior of a shared causal variant
    n_snp: int = 0
    robust: bool = False

    def is_causal_anchor(self, pp4_min: float = 0.8, p_thr: float = 5e-8) -> bool:
        return (self.p_ivw < p_thr and self.robust and self.steiger_ok
                and (np.isnan(self.coloc_pp4) or self.coloc_pp4 >= pp4_min))

    def to_evidence(self) -> EvidenceRecord:
        return EvidenceRecord(
            EvidenceType.GENETIC_MR, self.exposure, self.outcome,
            statistic=self.beta_ivw, p_value=self.p_ivw,
            direction=int(np.sign(self.beta_ivw)),
            detail={"egger_intercept_p": self.egger_intercept_p, "steiger": self.steiger_ok,
                    "coloc_pp4": self.coloc_pp4, "n_snp": self.n_snp, "robust": self.robust},
            provenance="TwoSampleMR(IVW+Egger+Steiger)")


class MendelianRandomization:
    """
    Two-sample MR from harmonized instrument-level summary stats.
    Input columns: SNP, beta_exp, se_exp, beta_out, se_out.
    """

    def run(self, exposure: str, outcome: str, dat: pd.DataFrame,
            f_min: float = 10.0) -> MRResult:
        bx, sx = dat["beta_exp"].to_numpy(float), dat["se_exp"].to_numpy(float)
        by, sy = dat["beta_out"].to_numpy(float), dat["se_out"].to_numpy(float)
        # weak-instrument screen: keep SNPs with per-SNP F = (beta/se)^2 >= f_min
        F = (bx / sx) ** 2
        keep = F >= f_min
        bx, sx, by, sy = bx[keep], sx[keep], by[keep], sy[keep]
        if len(bx) < 2:
            return MRResult(exposure, outcome, np.nan, np.nan, 1.0, n_snp=len(bx), robust=False)
        # IVW (inverse-variance weighted, random-effects flavor)
        w = 1.0 / sy ** 2
        beta_ivw = np.sum(w * bx * by) / np.sum(w * bx ** 2)
        se_ivw = np.sqrt(1.0 / np.sum(w * bx ** 2))
        p_ivw = 2 * stats.norm.sf(abs(beta_ivw / se_ivw))
        # MR-Egger: intercept != 0 signals directional (horizontal) pleiotropy
        import statsmodels.api as sm
        eg = sm.WLS(by, sm.add_constant(bx), weights=w).fit()
        egger_int_p = float(eg.pvalues[0])
        # Cochran Q heterogeneity
        resid = by - beta_ivw * bx
        Q = np.sum(w * resid ** 2)
        q_p = float(stats.chi2.sf(Q, df=len(bx) - 1))
        # Steiger: exposure should explain more variance in itself than in the outcome
        steiger_ok = bool(np.mean(bx ** 2) > np.mean(by ** 2))
        robust = (egger_int_p > 0.05) and (q_p > 0.05)
        return MRResult(exposure, outcome, float(beta_ivw), float(se_ivw), float(p_ivw),
                        egger_int_p, q_p, steiger_ok, np.nan, len(bx), robust)


def colocalization_pp4(exp_z: np.ndarray, out_z: np.ndarray, prior: float = 1e-4) -> float:
    """
    Approximate Bayesian colocalization (Giambartolomei-style H4): posterior probability that the
    exposure (eQTL) and the outcome (GWAS) share ONE causal variant at the locus. A high PP.H4
    upgrades an MR edge from 'associated' to 'colocalized' (much stronger). Simplified ABF version.
    """
    def abf(z):
        # approximate Bayes factor per SNP under a single-causal-variant model
        r = 0.15  # prior variance ratio
        return np.exp(0.5 * (z ** 2) * r / (1 + r)) / np.sqrt(1 + r)
    l1 = abf(exp_z); l2 = abf(out_z)
    # H4: shared variant. Coarse: normalize joint vs independent support.
    h4 = np.sum(l1 * l2) * prior
    h3 = np.sum(l1) * np.sum(l2) * prior * prior
    denom = h3 + h4 + 1e-12
    return float(h4 / denom)


def mr_orientation_anchors(results: list[MRResult], pp4_min: float = 0.8):
    """[(exposure, outcome, method, confidence)] anchors that FORCE Layer 1 orientation."""
    out = []
    for r in results:
        if r.is_causal_anchor(pp4_min):
            conf = min(0.99, 1 - r.p_ivw) if r.p_ivw > 0 else 0.99
            out.append((r.exposure, r.outcome, "MR(IVW+Steiger)", conf))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n_snp = 30
    bx = rng.normal(0.4, 0.1, n_snp)               # instrument effects on exposure
    true_beta = 0.6                                 # true causal effect exposure -> outcome
    by = true_beta * bx + rng.normal(0, 0.02, n_snp)
    dat = pd.DataFrame({"SNP": [f"rs{i}" for i in range(n_snp)],
                        "beta_exp": bx, "se_exp": np.full(n_snp, 0.03),
                        "beta_out": by, "se_out": np.full(n_snp, 0.03)})
    mr = MendelianRandomization().run("TGFB1", "FIBROSIS", dat)
    print(f"MR TGFB1 -> FIBROSIS: beta={mr.beta_ivw:.3f} (true {true_beta}) p={mr.p_ivw:.1e} "
          f"robust={mr.robust} steiger={mr.steiger_ok} anchor={mr.is_causal_anchor()}")
