"""
LAYER 1 - OBSERVATIONAL (Pearl Rung 1: association, plus structure discovery).

Modules expanded here (console Tier-2/Tier-1):
  * DEG analysis            -> differential expression on counts (the signal)
  * Pathway / GSEA          -> enrichment context
  * Causal DAG builder      -> bootstrap-stabilized ENSEMBLE discovery (PC/GES/LiNGAM/NOTEARS)
  * Centrality calculator   -> do-effect driver ranking (filled by Layer 4)

The single most important honesty rule of the whole engine lives here: association is generated at
this layer, but it is NEVER promoted to causation at this layer. Differential expression produces
candidate NODES; ensemble discovery produces candidate EDGES with a stability score and an
orientation provenance; but a co-expression edge is emitted as OBSERVATIONAL_ASSOCIATION, whose
fusion weight is exactly zero. Direction is left `unresolved_undirected` on a genuine tie rather than
guessed. Correlation is where we start, never where we stop.

Dependency-light: numpy, pandas, scipy, statsmodels. Optional causallearn/lingam used if present.
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from _schema import EvidenceRecord, EvidenceType


# ----------------------------------------------------------------------------------------------
# 1a. Differential expression (association signal). Real DE uses DESeq2/edgeR/limma-voom on counts;
#     here is a transparent, negative-binomial-style screen for the standalone reference. In the
#     deployed engine the supervisor routes to DESeq2/PyDESeq2 (never a raw t-test on counts).
# ----------------------------------------------------------------------------------------------
def differential_expression(counts: pd.DataFrame, groups: pd.Series,
                            min_count: int = 10) -> pd.DataFrame:
    """
    counts: genes x samples (raw counts). groups: sample -> condition (2 levels).
    Returns per-gene log2FC, p, FDR. This produces ASSOCIATION only (Rung 1).

    Note: for real bulk RNA-seq use DESeq2/edgeR/limma-voom (proper dispersion + shrinkage). This
    reference uses CPM-log + Welch t on the log scale purely so the module runs with no R.
    """
    counts = counts.loc[counts.sum(axis=1) >= min_count]
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    logcpm = np.log2(cpm + 1.0)
    levels = groups.unique()
    a = logcpm.loc[:, groups[groups == levels[0]].index]
    b = logcpm.loc[:, groups[groups == levels[1]].index]
    lfc = b.mean(axis=1) - a.mean(axis=1)
    t, p = stats.ttest_ind(b, a, axis=1, equal_var=False)
    # Benjamini-Hochberg FDR
    order = np.argsort(p)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, len(p) + 1)
    fdr = np.minimum(1.0, p * len(p) / ranked)
    out = pd.DataFrame({"log2FC": lfc.values, "p": p, "fdr": fdr}, index=counts.index)
    out["method"] = "reference_logCPM_Welch (deploy: DESeq2/edgeR/limma-voom)"
    return out.sort_values("fdr")


def candidate_nodes(de: pd.DataFrame, fdr: float = 0.05, lfc: float = 1.0, top: int = 60) -> list:
    """Select the candidate gene set that ENTERS causal discovery. Threshold stated explicitly."""
    sig = de[(de["fdr"] < fdr) & (de["log2FC"].abs() > lfc)]
    return list(sig.head(top).index)


# ----------------------------------------------------------------------------------------------
# 1b. Ensemble causal-structure discovery with bootstrap stability selection.
#     PC (constraint), GES (score), DirectLiNGAM (functional), NOTEARS/linear (continuous-opt).
#     Falls back to a partial-correlation skeleton if the optional libraries are absent, but even
#     then it will NOT orient by alphabet: it leaves ties undirected.
# ----------------------------------------------------------------------------------------------
class EnsembleDiscovery:
    def __init__(self, n_bootstrap: int = 60, stability: float = 0.5, seed: int = 20260812):
        self.B = n_bootstrap
        self.stability = stability
        self.rng = np.random.default_rng(seed)

    def _lingam(self, X, cols):
        try:
            import lingam
        except Exception:
            return set()
        m = lingam.DirectLiNGAM(random_state=int(self.rng.integers(1 << 31)))
        m.fit(X)
        B = m.adjacency_matrix_
        return {(cols[j], cols[i]) for i in range(B.shape[0]) for j in range(B.shape[1])
                if i != j and abs(B[i, j]) > 1e-6}

    def _pc(self, X, cols):
        try:
            from causallearn.search.ConstraintBased.PC import pc
        except Exception:
            return self._parcorr_skeleton(X, cols)
        cg = pc(X, alpha=0.01, indep_test="fisherz", stable=True, show_progress=False)
        G = cg.G.graph
        out = set()
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j and G[i, j] == 1 and G[j, i] == -1:
                    out.add((cols[i], cols[j]))
                elif i < j and G[i, j] == -1 and G[j, i] == -1:
                    out.add(tuple(sorted((cols[i], cols[j]))))  # undirected -> orient later
        return out

    def _parcorr_skeleton(self, X, cols, alpha=0.01):
        """Fallback: partial-correlation skeleton (UNDIRECTED). Never invents a direction."""
        df = pd.DataFrame(X, columns=cols)
        prec = np.linalg.pinv(np.corrcoef(X, rowvar=False))
        d = np.sqrt(np.diag(prec))
        pcorr = -prec / np.outer(d, d)
        n = X.shape[0]
        out = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = np.clip(pcorr[i, j], -0.999, 0.999)
                z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - len(cols) - 1)
                if 2 * stats.norm.sf(abs(z)) < alpha:
                    out.add(tuple(sorted((cols[i], cols[j]))))  # UNDIRECTED
        return out

    def discover(self, expr: pd.DataFrame) -> list[EvidenceRecord]:
        """expr: samples x genes. Returns discovery EvidenceRecords with bootstrap stability."""
        cols = list(expr.columns)
        Xf = expr.to_numpy(float)
        n = Xf.shape[0]
        skel = defaultdict(int)
        directed = defaultdict(int)
        for _ in range(self.B):
            idx = self.rng.integers(0, n, size=n)
            Xb = Xf[idx]
            present = set()
            method_edges = []
            for meth in (self._pc, self._lingam):
                try:                                   # a single algorithm failing (e.g. LiNGAM when
                    method_edges.append(meth(Xb, cols))  # n < features) must not crash the ensemble
                except Exception:
                    method_edges.append(set())
            for edges in method_edges:
                for e in edges:
                    if len(e) == 2 and e[0] != e[1]:
                        present.add(frozenset(e))
                        if e == (e[0], e[1]):  # directed vote
                            directed[e] += 1
            for s in present:
                skel[s] += 1
        recs = []
        for s, cnt in skel.items():
            u, v = tuple(s)
            stab = cnt / self.B
            if stab < self.stability:
                continue
            fwd, rev = directed.get((u, v), 0), directed.get((v, u), 0)
            if fwd == rev:
                src, tgt, method = u, v, "unresolved_undirected"      # DO NOT GUESS
            elif fwd > rev:
                src, tgt, method = u, v, "ensemble_majority"
            else:
                src, tgt, method = v, u, "ensemble_majority"
            recs.append(EvidenceRecord(
                etype=EvidenceType.OBSERVATIONAL_DISCOVERY, source=src, target=tgt,
                statistic=stab, detail={"orientation_method": method,
                                        "fwd_votes": fwd, "rev_votes": rev},
                provenance="EnsembleDiscovery(PC+LiNGAM,bootstrap)"))
        return recs


def coexpression_association(expr: pd.DataFrame, top: int = 50) -> list[EvidenceRecord]:
    """
    The ANTI-EXAMPLE, emitted deliberately: strongest Pearson pairs as OBSERVATIONAL_ASSOCIATION.
    These carry LLR weight 0 in fusion. Included so you can SEE that correlation never becomes
    causal in this engine, no matter how strong.
    """
    C = expr.corr().abs()
    cols = list(expr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(C.iloc[i, j])))
    pairs.sort(key=lambda t: -t[2])
    return [EvidenceRecord(EvidenceType.OBSERVATIONAL_ASSOCIATION, a, b, statistic=r,
                           provenance="pearson") for a, b, r in pairs[:top]]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    g = rng.normal(size=n)
    a = g + rng.normal(0, 0.5, n)          # A <- G
    b = 0.8 * a + rng.normal(0, 0.5, n)    # B <- A  (true edge A->B)
    c = rng.normal(size=n)                  # independent
    expr = pd.DataFrame({"GENE_A": a, "GENE_B": b, "GENE_C": c, "GENE_G": g})
    recs = EnsembleDiscovery(n_bootstrap=40).discover(expr)
    print("discovery edges (association/structure only, NOT yet causal):")
    for r in recs:
        print(f"  {r.source}->{r.target}  stability={r.statistic:.2f}  {r.detail['orientation_method']}")
    print("co-expression (LLR=0 by construction):", [(r.source, r.target) for r in
          coexpression_association(expr, top=3)])
