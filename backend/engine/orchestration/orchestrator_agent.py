"""
ORCHESTRATOR AGENT - turns a natural-language request (Causion) into a plan, routes to the
supervisor, and only AFTER the verdict is frozen lets a language model write the narrative.

The governance contract, enforced in code:
  1. intent classification and module-chain planning are the orchestrator's job (it decides WHAT
     runs), NOT the causal decision;
  2. the CausalitySupervisor computes the verdict deterministically (identification, estimation,
     refutation, fusion, calibration);
  3. a language model may then summarize the FROZEN, confirmed edges, but a grounding check deletes
     any sentence naming a gene not in the confirmed evidence. The LLM cannot invent, re-rank, or
     re-score a single edge.

This is the precise sense in which "the agents handle the causality proof": they orchestrate and
narrate; they never let a language model touch a load-bearing number.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules_registry import REGISTRY, CAUSAL_CHAIN  # noqa: E402
from supervisor_agent import CausalitySupervisor  # noqa: E402


# ---- intent classification (rule-based reference; deployment may use an LLM for PARSING only) ----
INTENT_PATTERNS = {
    "causal_driver_discovery": r"\b(caus\w+|driver|drive|mechanis\w+|target)\b",
    "patient_stratification": r"\b(patient|stratif\w+|responder|precision)\b",
    "perturbation_prediction": r"\b(knockout|knockdown|perturb\w+|silence|over-?express)\b",
    "differential_expression": r"\b(differential|deg|up-?regulat\w+|down-?regulat\w+)\b",
}


def classify_intent(query: str) -> str:
    q = query.lower()
    for intent, pat in INTENT_PATTERNS.items():
        if re.search(pat, q):
            return intent
    return "causal_driver_discovery"


def plan_module_chain(intent: str) -> list[str]:
    """Return the ordered module chain for an intent. Causal intents traverse all four layers."""
    chain = []
    for layer, mods in CAUSAL_CHAIN:
        chain.extend(mods)
    if intent == "patient_stratification":
        chain.append("insilico_perturb_patient")
    return [m for m in chain if m in REGISTRY]


# ---- narrative LLM (walled off from scoring) ----
def _ground_check(text: str, allowed_genes: set) -> bool:
    toks = {t.strip(".,;:()").upper() for t in text.replace("->", " ").split()}
    reserved = {"MR", "DAG", "CCS", "AIPW", "IVW", "CRISPR", "RNA", "DNA", "FDR", "AI", "LLM", "SCM"}
    foreign = {t for t in toks if t.isalpha() and len(t) >= 2 and t not in allowed_genes
               and t not in reserved and t.isupper()}
    return len(foreign) == 0


def narrate(confirmed_edges, llm_complete=None) -> dict:
    """
    Produce a grounded narrative over FROZEN confirmed edges. If no LLM is injected, emit a
    deterministic template summary (still fully grounded). The LLM, if present, may only rephrase;
    a grounding check drops ungrounded sentences.
    """
    allowed = {e.source for e in confirmed_edges} | {e.target for e in confirmed_edges}
    facts = [f"{e.source} -> {e.target} (CCS {e.ccs:.2f}, {e.ccs_tier})" for e in confirmed_edges]
    template = ("Confirmed causal drivers, each identified, refuted and calibrated: "
                + "; ".join(facts) + ".") if facts else "No edge passed the honest causal gate."
    if llm_complete is None:
        return {"summary": template, "grounded": True, "n_confirmed": len(confirmed_edges)}
    raw = llm_complete(system="Summarize ONLY these frozen causal edges; invent nothing.",
                       user=template)
    return {"summary": raw if _ground_check(raw, allowed) else template,
            "grounded": _ground_check(raw, allowed), "n_confirmed": len(confirmed_edges)}


class Orchestrator:
    def __init__(self, calibrator=None):
        self.supervisor = CausalitySupervisor(calibrator=calibrator)

    def handle(self, query: str, expr, *, mr_results=None, temporal_ts=None,
               signor_edges=None, ncrna_anchors=None, llm_complete=None) -> dict:
        intent = classify_intent(query)
        chain = plan_module_chain(intent)
        result = self.supervisor.run(expr, mr_results=mr_results, temporal_ts=temporal_ts,
                                     signor_edges=signor_edges, ncrna_anchors=ncrna_anchors)
        narrative = narrate(result["confirmed"], llm_complete=llm_complete)
        return {"intent": intent, "planned_chain": chain,
                "confirmed": [e.to_row() for e in result["confirmed"]],
                "drivers": result["drivers"], "narrative": narrative}


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 400
    conf = rng.normal(size=n)
    tgfb1 = 0.6 * conf + rng.normal(0, 0.5, n)
    fn1 = 1.2 * tgfb1 + 0.8 * conf + rng.normal(0, 0.5, n)
    expr = pd.DataFrame({"CONF": conf, "TGFB1": tgfb1, "FN1": fn1, "NOISE": rng.normal(size=n)})
    out = Orchestrator().handle("find the causal drivers of fibrosis", expr,
                                signor_edges={("TGFB1", "FN1")})
    print("intent:", out["intent"])
    print("planned chain:", out["planned_chain"][:6], "...")
    print("narrative:", out["narrative"]["summary"])
