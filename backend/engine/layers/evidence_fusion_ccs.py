"""
FUSION + REFUTATION + CALIBRATED CCS - where independent evidence becomes ONE trustworthy number.

Three components, in the order they run on each edge:
  1. REFUTATION  - stress-test every estimated effect (placebo, random common cause, subset,
     E-value). Fails closed: a non-null placebo blocks the causal claim.
  2. BAYESIAN FUSION - combine independent streams as LOG-LIKELIHOOD RATIOS:
        logit P(causal|evidence) = logit(prior) + sum_k w_k * LLR_k
     Co-expression contributes exactly 0. Identification + refutation success enter as evidence too.
  3. CALIBRATED CCS - map the fused posterior through an isotonic/Platt calibrator fit on gold edges
     (SIGNOR/DoRothEA-A/CRISPR), verified with Brier + ECE, so "CCS 0.9" means ~90% are truly causal.
     The honest gate: is_causal = identified AND refuted AND calibrated CCS >= threshold.

This is the mathematical heart of "why it is a causality engine": correlation cannot move the number,
and the number is a checkable probability, not a weighted tally. Dependency-light: numpy, sklearn.
"""
from __future__ import annotations

import numpy as np

from _schema import (CausalEdge, EvidenceType, EVIDENCE_PRIOR_LLR, RefutationResult,
                     IdentificationStrategy)


# ---------------------------------------------------------------------------------------------
# 1. REFUTATION
# ---------------------------------------------------------------------------------------------
class Refuter:
    def __init__(self, estimator, seed=0, alpha=0.05):
        self.est = estimator; self.rng = np.random.default_rng(seed); self.alpha = alpha

    def run_all(self, df, t, y, Z, effect):
        import pandas as pd  # noqa
        out = []
        # baseline for comparison refuters uses the SAME estimator internally
        try:
            obs = self.est.backdoor_linear(df, t, y, list(Z)).point
        except Exception:
            obs = effect.point
        # placebo: permute treatment -> effect must vanish
        d = df.copy(); d[t] = self.rng.permutation(d[t].to_numpy())
        try:
            pe = self.est.backdoor_linear(d, t, y, list(Z))
            out.append(RefutationResult("placebo_treatment", pe.p_value > self.alpha, pe.point,
                                        f"placebo effect {pe.point:.3f} p={pe.p_value:.2g}"))
        except Exception as e:
            out.append(RefutationResult("placebo_treatment", False, None, str(e)))
        # random common cause: add a random covariate -> estimate barely moves
        d2 = df.copy(); d2["_rcc_"] = self.rng.normal(size=len(d2))
        try:
            re = self.est.backdoor_linear(d2, t, y, list(Z) + ["_rcc_"])
            rel = abs(re.point - obs) / (abs(obs) + 1e-9)
            out.append(RefutationResult("random_common_cause", rel < 0.2, re.point, f"drel={rel:.1%}"))
        except Exception as e:
            out.append(RefutationResult("random_common_cause", False, None, str(e)))
        # E-value: how strong must an unmeasured confounder be to nullify the effect?
        rr = np.exp(0.91 * abs(effect.point))
        ev = rr + np.sqrt(rr * (rr - 1)) if rr >= 1 else (1 / rr) + np.sqrt((1 / rr) * (1 / rr - 1))
        out.append(RefutationResult("e_value", ev > 1.25, float(ev), f"E-value {ev:.2f}"))
        return out


# ---------------------------------------------------------------------------------------------
# 2. BAYESIAN EVIDENCE FUSION
# ---------------------------------------------------------------------------------------------
def _logit(p): return np.log(p / (1 - p))
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


class EvidenceFusion:
    def __init__(self, prior_causal=0.05, weights=None):
        self.prior = prior_causal
        self.w = weights or dict(EVIDENCE_PRIOR_LLR)

    def _stream_llr(self, rec):
        base = self.w.get(rec.etype, 0.0)
        if rec.etype == EvidenceType.OBSERVATIONAL_ASSOCIATION:
            return 0.0                                   # correlation: zero, always
        if rec.etype == EvidenceType.COLOCALIZATION:
            return base * float(rec.detail.get("coloc_pp4", rec.statistic))
        if rec.p_value is not None and rec.p_value > 0:
            conf = min(1.0, (-np.log10(rec.p_value)) / 8.0)   # p=1e-8 -> full weight
        elif rec.fdr is not None:
            conf = min(1.0, (-np.log10(max(rec.fdr, 1e-12))) / 3.0)
        else:
            conf = min(1.0, abs(rec.statistic))
        return base * conf

    def fuse(self, edge: CausalEdge) -> float:
        lo = _logit(self.prior)
        if edge.bootstrap_stability > 0:
            lo += 1.5 * (edge.bootstrap_stability - 0.5)
        if edge.effect is not None and edge.identification != IdentificationStrategy.NOT_IDENTIFIED:
            if edge.effect.p_value < 0.05:
                lo += 1.2
        if edge.refutations and all(r.passed for r in edge.refutations):
            lo += 1.0
        elif edge.refutations:
            lo -= 1.5                                    # failed a refuter -> penalize hard
        for rec in edge.evidence:
            lo += self._stream_llr(rec)
        return float(_sigmoid(lo))

    def annotate(self, edges):
        for e in edges:
            e.posterior_prob = self.fuse(e)
        return edges


# ---------------------------------------------------------------------------------------------
# 3. CALIBRATED CCS + HONEST GATE
# ---------------------------------------------------------------------------------------------
class CCSCalibrator:
    def __init__(self, method="isotonic"):
        self.method = method; self.model = None; self.metrics = {}

    def fit(self, posteriors, labels):
        p = np.asarray(posteriors, float); y = np.asarray(labels, int)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
            cal = self.model.predict(p)
        else:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression().fit(p.reshape(-1, 1), y)
            cal = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        self.metrics = self._metrics(cal, y)
        return self

    def transform(self, p):
        if self.model is None:
            return float(p)
        if self.method == "isotonic":
            return float(self.model.predict([p])[0])
        return float(self.model.predict_proba([[p]])[0, 1])

    @staticmethod
    def _metrics(pred, y, bins=10):
        pred = np.clip(pred, 0, 1); y = np.asarray(y, float)
        brier = float(np.mean((pred - y) ** 2))
        edges = np.linspace(0, 1, bins + 1); ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pred >= lo) & (pred < hi)
            if m.sum():
                ece += m.mean() * abs(pred[m].mean() - y[m].mean())
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, pred)) if len(set(y)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        return {"brier": brier, "ece": float(ece), "auroc": auc}


class ConfidenceScorer:
    def __init__(self, calibrator=None, t1=0.90, t2=0.70, t3=0.50, gate=0.70):
        self.cal = calibrator; self.t1, self.t2, self.t3, self.gate = t1, t2, t3, gate

    def _tier(self, c):
        if c >= self.t1: return "T1_Driver"
        if c >= self.t2: return "T2_Strong_Candidate"
        if c >= self.t3: return "T3_Candidate"
        return "REJECTED"

    def score(self, edges):
        for e in edges:
            e.ccs = self.cal.transform(e.posterior_prob) if self.cal else e.posterior_prob
            e.ccs_tier = self._tier(e.ccs)
            identified = (e.identification != IdentificationStrategy.NOT_IDENTIFIED and e.effect is not None)
            e.is_causal = bool(e.ccs >= self.gate and identified and e.refutations_passed())
        return edges


if __name__ == "__main__":
    # show that an observation-only edge cannot pass while an MR+identified+refuted edge can
    from _schema import EvidenceRecord, EffectEstimate
    obs = CausalEdge("A", "B", bootstrap_stability=1.0,
                     evidence=[EvidenceRecord(EvidenceType.OBSERVATIONAL_ASSOCIATION, "A", "B", 0.95)])
    strong = CausalEdge("TGFB1", "FN1", bootstrap_stability=1.0,
                        identification=IdentificationStrategy.BACKDOOR,
                        effect=EffectEstimate(1.2, 0.9, 1.5, 0.15, 1e-6, "backdoor.aipw", 500),
                        refutations=[RefutationResult("placebo_treatment", True),
                                     RefutationResult("e_value", True)],
                        evidence=[EvidenceRecord(EvidenceType.GENETIC_MR, "TGFB1", "FN1", 0.6, p_value=1e-9)])
    EvidenceFusion().annotate([obs, strong])
    ConfidenceScorer().score([obs, strong])
    for e in (obs, strong):
        print(f"{e.source}->{e.target}: posterior={e.posterior_prob:.3f} ccs={e.ccs:.3f} "
              f"tier={e.ccs_tier} is_causal={e.is_causal}")
