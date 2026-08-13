"""
LAYER 2 - TEMPORAL (time-ordered causal evidence).

Module expanded: Temporal Granger analysis (+ Temporal analysis bulk).

Cause precedes effect. But naive pairwise Granger is a trap: it confuses a common cause for a
direct one, and it computes invalid p-values on ultra-short series. This layer therefore does two
honest things the audited predecessor did not:
  1. it CONDITIONS on the rest of the system (multivariate VAR / PCMCI+), removing common-cause
     confounding, rather than testing pairs in isolation;
  2. it REFUSES to run below a minimum number of timepoints (default 8) instead of manufacturing
     a p-value from 4 points.

Output: TEMPORAL_PRECEDENCE EvidenceRecords (lagged, FDR-controlled), which also serve as an
orientation anchor for Layer 1 discovery. Dependency-light: statsmodels (VAR). tigramite used if present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _schema import EvidenceRecord, EvidenceType


class TemporalCausal:
    def __init__(self, min_timepoints: int = 8, tau_max: int = 3, fdr_alpha: float = 0.05):
        self.min_tp = min_timepoints
        self.tau_max = tau_max
        self.fdr_alpha = fdr_alpha

    def run(self, ts: pd.DataFrame) -> list[EvidenceRecord]:
        """ts: ordered timepoints (rows) x genes (cols)."""
        if len(ts) < self.min_tp:
            # Honest refusal: too few points for a valid time-lagged causal test.
            return []
        try:
            return self._pcmci(ts)
        except Exception:
            return self._conditional_granger(ts)

    def _pcmci(self, ts: pd.DataFrame) -> list[EvidenceRecord]:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        df = pp.DataFrame(ts.to_numpy(float), var_names=list(ts.columns))
        pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
        res = pcmci.run_pcmciplus(tau_max=self.tau_max, pc_alpha=0.05)
        q = pcmci.get_corrected_pvalues(p_matrix=res["p_matrix"], fdr_method="fdr_bh")["q_matrix"]
        names = list(ts.columns)
        out = []
        for i, s in enumerate(names):
            for j, t in enumerate(names):
                if i == j:
                    continue
                for tau in range(1, self.tau_max + 1):
                    if q[i, j, tau] < self.fdr_alpha:
                        out.append(EvidenceRecord(
                            EvidenceType.TEMPORAL_PRECEDENCE, s, t,
                            statistic=float(res["val_matrix"][i, j, tau]),
                            p_value=float(res["p_matrix"][i, j, tau]), fdr=float(q[i, j, tau]),
                            detail={"lag": tau, "method": "PCMCI+"}, provenance="tigramite"))
        return out

    def _conditional_granger(self, ts: pd.DataFrame) -> list[EvidenceRecord]:
        """Conditional Granger through a full VAR: conditions on ALL genes, not just the pair."""
        from statsmodels.tsa.api import VAR
        from statsmodels.stats.multitest import multipletests
        d = ts.diff().dropna()                                   # difference for stationarity
        maxlag = min(self.tau_max, max(1, len(d) // 3))
        model = VAR(d).fit(maxlags=maxlag)
        names = list(ts.columns)
        pairs, pvals = [], []
        for t in names:
            for s in names:
                if s == t:
                    continue
                try:
                    res = model.test_causality(t, [s], kind="f")  # conditioned on the rest
                    pairs.append((s, t)); pvals.append(res.pvalue)
                except Exception:
                    pass
        if not pvals:
            return []
        q = multipletests(pvals, alpha=self.fdr_alpha, method="fdr_bh")[1]
        return [EvidenceRecord(EvidenceType.TEMPORAL_PRECEDENCE, s, t, statistic=1.0,
                               p_value=p, fdr=float(qv),
                               detail={"method": "conditional_granger_VAR"},
                               provenance="statsmodels.VAR")
                for (s, t), p, qv in zip(pairs, pvals, q) if qv < self.fdr_alpha]


def temporal_orientation_anchors(records: list[EvidenceRecord]):
    """[(source, target, confidence)] anchors for Layer 1 discovery from FDR-sig precedence."""
    return [(r.source, r.target, 1.0 - (r.fdr if r.fdr is not None else 0.05)) for r in records]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T = 40
    x = np.zeros(T); y = np.zeros(T); z = np.zeros(T)
    for k in range(1, T):
        x[k] = 0.5 * x[k - 1] + rng.normal(0, 1)
        y[k] = 0.6 * x[k - 1] + 0.3 * y[k - 1] + rng.normal(0, 1)   # X(t-1) -> Y(t)
        z[k] = 0.4 * z[k - 1] + rng.normal(0, 1)
    ts = pd.DataFrame({"X": x, "Y": y, "Z": z})
    recs = TemporalCausal().run(ts)
    print("temporal precedence edges (conditioned on the system):")
    for r in recs:
        print(f"  {r.source}->{r.target}  fdr={r.fdr:.3g}  {r.detail}")
    print("refusal check (T=5):", TemporalCausal().run(ts.head(5)))
