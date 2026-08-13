"""
SUPERVISOR AGENT (causality) - sequences the four layers and ENFORCES the honest gate.

This is the agent that "handles the causality proof". Its intelligence is orchestration; the proof
is deterministic statistics in the layer modules it calls. It:
  1. runs the genuinely-causal streams first (MR, temporal) to get orientation anchors;
  2. runs bootstrap discovery, oriented by those anchors (never by alphabet);
  3. builds an acyclic DAG, then for each edge runs identification -> estimation -> refutation;
  4. attaches all evidence streams to each edge;
  5. fuses evidence (Bayesian LLR) and scores a CALIBRATED CCS;
  6. applies the honest gate: is_causal = identified AND refuted AND CCS >= threshold.

Crucially, no language model is anywhere in this path. The verdict is computed. A narrative LLM only
runs afterwards, over the FROZEN edges (see orchestrator_agent.py).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layers"))

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from _schema import CausalEdge, EvidenceType, IdentificationStrategy  # noqa: E402
from layer1_observational import EnsembleDiscovery  # noqa: E402
from layer2_temporal import TemporalCausal, temporal_orientation_anchors  # noqa: E402
from layer3_genetic import mr_orientation_anchors  # noqa: E402
from layer4_interventional import Identifier, Estimator, SCM, InSilicoPerturbation  # noqa: E402
from evidence_fusion_ccs import EvidenceFusion, ConfidenceScorer, CCSCalibrator, Refuter  # noqa: E402


class CausalitySupervisor:
    def __init__(self, calibrator: CCSCalibrator | None = None, n_bootstrap: int = 40,
                 stability: float = 0.5, seed: int = 20260812):
        self.calibrator = calibrator
        self.discovery = EnsembleDiscovery(n_bootstrap=n_bootstrap, stability=stability, seed=seed)
        self.temporal = TemporalCausal()
        self.estimator = Estimator(seed=seed)
        self.refuter = Refuter(self.estimator, seed=seed)
        self.seed = seed

    # ---- the 9-step causal flow ----
    def run(self, expr: pd.DataFrame, *, mr_results=None, temporal_ts=None,
            signor_edges=None, ncrna_anchors=None, depmap=None) -> dict:
        # 1-2. anchors from genuinely-causal streams (MR, temporal); priors from SIGNOR/ncRNA
        mr_anchors = mr_orientation_anchors(mr_results or [])
        temp_records = self.temporal.run(temporal_ts) if temporal_ts is not None else []
        temp_anchors = temporal_orientation_anchors(temp_records)
        anchors = {}
        for (u, v, m, c) in mr_anchors:
            anchors[(u, v)] = (m, c)
        for (u, v, c) in temp_anchors:
            anchors.setdefault((u, v), ("temporal", c))
        for (u, v, m, c) in (ncrna_anchors or []):
            anchors.setdefault((u, v), (m, c))
        required = set(signor_edges or [])

        # 3. discovery, then re-orient by anchors/priors (anchors win over ensemble vote)
        disc = self.discovery.discover(expr)
        edges: dict[tuple, CausalEdge] = {}
        for r in disc:
            s, t = r.source, r.target
            method = r.detail.get("orientation_method", "ensemble")
            if (t, s) in anchors and (s, t) not in anchors:
                s, t = t, s
            if (s, t) in anchors:
                method = f"anchored:{anchors[(s, t)][0]}"
            e = edges.setdefault((s, t), CausalEdge(s, t, orientation_method=method,
                                                    bootstrap_stability=r.statistic))
            e.evidence.append(r)
        for (u, v) in required:                     # SIGNOR required directed edges
            edges.setdefault((u, v), CausalEdge(u, v, orientation_method="signor_required",
                                                bootstrap_stability=0.6))

        # build acyclic DAG from oriented (non-tie) edges
        dag = nx.DiGraph(); dag.add_nodes_from(expr.columns)
        for (s, t), e in edges.items():
            if e.orientation_method != "unresolved_undirected":
                dag.add_edge(s, t)
        dag = self._acyclic(dag, edges)

        # 4-6. identify -> estimate -> refute per surviving oriented edge
        ident = Identifier(dag)
        for (s, t), e in edges.items():
            if not dag.has_edge(s, t):
                continue
            strat, Z = ident.identify(s, t)
            e.identification = strat; e.adjustment_set = sorted(Z)
            if strat != IdentificationStrategy.NOT_IDENTIFIED:
                e.effect = self.estimator.estimate(expr, s, t, strat, Z)
                if e.effect is not None:
                    e.refutations = self.refuter.run_all(expr, s, t, e.adjustment_set, e.effect)
            # attach MR evidence
            for r in (mr_results or []):
                if r.exposure == s and r.outcome == t:
                    e.evidence.append(r.to_evidence())
            # attach temporal evidence
            for tr in temp_records:
                if tr.source == s and tr.target == t:
                    e.evidence.append(tr)

        # 6b. interventional cross-check via in-silico perturbation on the fitted SCM
        insilico = {}
        try:
            scm = SCM.fit(expr, dag)
            pert = InSilicoPerturbation(scm, seed=self.seed)
            for (s, t), e in edges.items():
                if dag.has_edge(s, t) and e.effect is not None:
                    insilico[(s, t)] = pert.knockout(expr, s, [t]).get(t)
        except Exception:
            pass

        # 7-8. Bayesian fusion -> calibrated CCS
        elist = list(edges.values())
        EvidenceFusion().annotate(elist)
        ConfidenceScorer(self.calibrator).score(elist)

        # 9. driver scores from identified, calibrated do-effects
        drivers = {}
        for e in elist:
            if e.is_causal and e.effect is not None:
                drivers[e.source] = drivers.get(e.source, 0.0) + abs(e.effect.point) * e.ccs

        return {"edges": elist,
                "confirmed": [e for e in elist if e.is_causal],
                "drivers": dict(sorted(drivers.items(), key=lambda kv: -kv[1])),
                "insilico": insilico,
                "dag": dag}

    @staticmethod
    def _acyclic(dag, edges):
        stab = {(e.source, e.target): e.bootstrap_stability for e in edges.values()}
        while not nx.is_directed_acyclic_graph(dag):
            cyc = nx.find_cycle(dag)
            weakest = min(cyc, key=lambda uv: stab.get((uv[0], uv[1]), 0.0))
            dag.remove_edge(weakest[0], weakest[1])
        return dag


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    n = 400
    conf = rng.normal(size=n)
    tgfb1 = 0.6 * conf + rng.normal(0, 0.5, n)
    fn1 = 1.2 * tgfb1 + 0.8 * conf + rng.normal(0, 0.5, n)
    noise = rng.normal(size=n)
    expr = pd.DataFrame({"CONF": conf, "TGFB1": tgfb1, "FN1": fn1, "NOISE": noise})
    res = CausalitySupervisor(n_bootstrap=30).run(expr, signor_edges={("TGFB1", "FN1")})
    print("confirmed causal edges:", [(e.source, e.target, round(e.ccs, 2)) for e in res["confirmed"]])
    print("drivers:", {k: round(v, 2) for k, v in res["drivers"].items()})
