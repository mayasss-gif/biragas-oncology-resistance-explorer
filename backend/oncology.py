#!/usr/bin/env python3
"""
Oncology resistance-pathway analysis for the BiRAGAS web app.

Data (backend/data/, built from the REAL cited source
BiRAGAS_Oncology_Failed_Therapies.xlsx + pathways.json):
  oncology_pathways.json      - 209 enriched entries (full medical context + per-entry cited genes)
  resistance_signatures.json  - 13 resistance-class gene signatures, grounded in the DB's own taxonomy

When a user picks one of the 209 pathways and uploads a DEG (or expression) file, we show, side by side:
  1. the pathway's real MEDICAL CONTEXT (cancer, failed drug, target, resistance mechanism, class,
     BiRAGAS pattern, trial/NCT, citation);
  2. a CLASS-SIGNATURE map: which genes of that resistance class's signature are differentially
     expressed in this sample (variable count, real stats), plus the entry's own cited genes;
  3. the overall DEG LANDSCAPE (volcano + ranked differential genes).

Nothing is fabricated: entries are verbatim from the cited DB; signatures are standard resistance
biology grounded in the DB taxonomy; the map uses the real fold-change / p-value in the uploaded file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
import live_data  # noqa: E402  (Ensembl / ChEMBL / AlphaFold / GTEx / cBioPortal / PubChem)

_PATHWAYS = json.load(open(os.path.join(DATA, "oncology_pathways.json")))
_SIGS = json.load(open(os.path.join(DATA, "resistance_signatures.json")))


def _ftype(g):
    s = str(g); low = s.lower()
    if low.startswith(("mir", "hsa-mir", "hsa-let")) or "-mir-" in low:
        return "miRNA"
    if "linc" in low or s.endswith(("-AS1", "-AS2")) or low.startswith("lnc"):
        return "lncRNA"
    return "mRNA"


def list_pathways():
    """Compact list for the dropdown, grouped by cancer site."""
    out = []
    for p in _PATHWAYS:
        out.append({
            "idx": p["idx"], "id": p["id"], "site": p.get("site") or p.get("cancer_type", ""),
            "group": p["group"], "cancer_type": p["cancer_type"], "drug": p["drug"],
            "resistance_class": p["resistance_class"], "pattern": p["biragas_pattern"],
            "label": f'{p["cancer_type"]} - {p["drug"]}  [{p["resistance_class"]}]',
        })
    return {"n": len(out), "pathways": out}


def get_pathway(idx):
    for p in _PATHWAYS:
        if int(p["idx"]) == int(idx):
            return p
    return None


def _signature(rc):
    s = _SIGS.get(rc, {"genes": [], "note": ""})
    return list(s.get("genes", [])), s.get("note", "")


def _hits(deg_by_gene: dict, genes, source):
    """Intersect a gene list with the patient's DEG table; return per-gene stats (real)."""
    hits = []
    for g in genes:
        row = deg_by_gene.get(g)
        if row is None:
            continue
        lfc = float(row["log2fc"])
        pa = row["padj"]
        sig = bool(pd.notna(pa) and float(pa) < 0.05 and abs(lfc) > 1)
        hits.append({"gene": g, "log2fc": round(lfc, 3),
                     "direction": "up" if lfc >= 0 else "down",
                     "padj": (None if pd.isna(pa) else float(pa)), "sig": sig, "source": source})
    hits.sort(key=lambda h: -abs(h["log2fc"]))
    return hits


def run_pathway_analysis(deg: pd.DataFrame, pathway: dict, kind: str = "deg"):
    """
    deg: standardized DEG table with columns gene, log2fc, pvalue, padj, direction.
    pathway: one entry dict from oncology_pathways.json.
    """
    deg = deg.copy()
    deg["gene"] = deg["gene"].astype(str)
    p_rank = deg["padj"].fillna(deg["pvalue"]).clip(lower=1e-300)
    deg["neglog10p"] = (-np.log10(p_rank)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    deg["score"] = deg["log2fc"].abs() * deg["neglog10p"]
    deg["sig"] = (deg["padj"].fillna(1.0) < 0.05) & (deg["log2fc"].abs() > 1)
    by_gene = {r["gene"]: r for _, r in deg.iterrows()}

    rc = pathway["resistance_class"]
    sig_genes, sig_note = _signature(rc)
    entry_genes = list(pathway.get("entry_genes", []))

    sig_hits = _hits(by_gene, sig_genes, "class")
    entry_extra = [g for g in entry_genes if g not in sig_genes]
    entry_hits = _hits(by_gene, entry_extra, "entry")
    hit_set = {h["gene"] for h in sig_hits} | {h["gene"] for h in entry_hits}
    focus_set = set(sig_genes) | set(entry_genes)

    # LIVE target intelligence for the resistance-pathway genes altered in this sample: Ensembl id,
    # existing drugs (ChEMBL + clinical phase), AlphaFold structure, GTEx expression, cBioPortal/PubChem.
    # Enrich the strongest-effect hits (bounded to keep the request responsive; cached).
    failed_drug = pathway.get("drug", "")
    intel_hits = sorted(sig_hits + entry_hits, key=lambda h: -abs(h["log2fc"]))[:6]
    target_intel = []
    for h in intel_hits:
        up = h["log2fc"] >= 0
        try:
            live = live_data.enrich_target(h["gene"])
        except Exception:
            live = None
        # RATIONAL REPLACEMENT: the failed drug hit its original target and failed via this resistance
        # mechanism; a drug that targets THIS altered resistance node is the rational replacement/add-on.
        # Most actionable when the node is over-expressed (up) and an approved inhibitor exists.
        drugs = ((live or {}).get("chembl") or {}).get("drugs") or []
        approved = [d for d in drugs if (d.get("max_phase") or 0) >= 4]
        clinical = [d for d in drugs if 0 < (d.get("max_phase") or 0) < 4]
        rep = approved or clinical
        replacement = None
        if rep:
            replacement = {"drugs": [d["name"] for d in rep[:3]],
                           "phase": rep[0].get("max_phase"),
                           "approved": bool(approved),
                           "actionable": bool(up and approved)}   # over-active node + approved inhibitor
        target_intel.append({
            "gene": h["gene"], "tier": ("signature" if h["source"] == "class" else "cited"),
            "direction": h["direction"], "log2fc": h["log2fc"], "padj": h.get("padj"),
            "ko_action": ("knock down / inhibit" if up else
                          "loss-of-function; restore or target a synthetic-lethal partner"),
            "failed_drug": failed_drug, "replacement": replacement, "live": live})
    n_druggable = sum(1 for t in target_intel
                      if (t.get("live") or {}).get("chembl", {}).get("max_phase") and
                      (t["live"]["chembl"]["max_phase"] or 0) >= 4)

    # volcano (cap for the browser) with signature / entry flags
    vdf = deg.sort_values("score", ascending=False).head(600)
    volcano = []
    for _, r in vdf.iterrows():
        g = r["gene"]
        pp = r["pvalue"]
        volcano.append({"feature": g, "type": _ftype(g), "log2fc": round(float(r["log2fc"]), 3),
                        "neglog10p": (None if pd.isna(pp) else round(float(-np.log10(max(float(pp), 1e-300))), 3)),
                        "sig": bool(r["sig"]),
                        "in_signature": g in set(sig_genes), "in_entry": g in set(entry_genes)})
    # always include the signature/entry hits in the volcano even if beyond top-600
    have = {v["feature"] for v in volcano}
    for g in focus_set:
        if g in by_gene and g not in have:
            r = by_gene[g]; pp = r["pvalue"]
            volcano.append({"feature": g, "type": _ftype(g), "log2fc": round(float(r["log2fc"]), 3),
                            "neglog10p": (None if pd.isna(pp) else round(float(-np.log10(max(float(pp), 1e-300))), 3)),
                            "sig": bool(r["sig"]), "in_signature": g in set(sig_genes), "in_entry": g in set(entry_genes)})

    # overall ranked differential genes (variable count: the significant ones, capped for display)
    sig_deg = deg[deg["sig"]].sort_values("score", ascending=False)
    top = (sig_deg if len(sig_deg) else deg.sort_values("score", ascending=False)).head(25)
    drivers = [{"gene": r["gene"], "log2fc": round(float(r["log2fc"]), 3),
                "direction": "up" if r["log2fc"] >= 0 else "down",
                "padj": (None if pd.isna(r["padj"]) else float(r["padj"])),
                "score": round(float(r["score"]), 2),
                "in_focus": r["gene"] in focus_set}
               for _, r in top.iterrows()]

    n_up = int((deg["direction"] == "up").sum()) if "direction" in deg else int((deg["log2fc"] >= 0).sum())
    n_dn = int(len(deg) - n_up)
    n_sig = int(deg["sig"].sum())
    sig_cov = round(len(sig_hits) / max(1, len(sig_genes)), 3) if sig_genes else 0.0

    return {
        "mode": "oncology",
        "input_kind": kind,
        "pathway": {
            "idx": pathway["idx"], "id": pathway.get("id", ""),
            "cancer_type": pathway["cancer_type"], "group": pathway["group"],
            "site": pathway.get("site", ""), "modality": pathway.get("modality", ""),
            "drug": pathway["drug"], "target_mechanism": pathway.get("target_mechanism", ""),
            "resistance_mechanism": pathway.get("resistance_mechanism", ""),
            "resistance_class": rc, "biragas_pattern": pathway.get("biragas_pattern", ""),
            "outcome": pathway.get("outcome", ""), "trial": pathway.get("trial", ""),
            "nct": pathway.get("nct", ""), "sponsor": pathway.get("sponsor", ""),
            "citation": pathway.get("citation", ""), "confidence": pathway.get("confidence", ""),
            "signature_note": sig_note, "signature_size": len(sig_genes),
            "signature_genes": sig_genes, "entry_genes": entry_genes,
        },
        "kpis": {
            "n_features": int(len(deg)), "n_up": n_up, "n_down": n_dn, "n_sig": n_sig,
            "signature_size": len(sig_genes), "signature_hits": len(sig_hits),
            "signature_coverage": sig_cov, "entry_hits": len(entry_hits),
            "pattern": pathway.get("biragas_pattern", ""), "n_druggable": n_druggable,
        },
        "signature_map": {"resistance_class": rc, "note": sig_note, "size": len(sig_genes),
                          "hits": sig_hits, "n_hits": len(sig_hits), "coverage": sig_cov},
        "entry_map": {"genes": entry_genes, "hits": entry_hits, "n_hits": len(entry_hits)},
        "target_intel": target_intel,
        "volcano": volcano,
        "drivers": drivers,
        "honest_note": (
            "This maps your uploaded differential expression onto the chosen resistance pathway. The "
            "medical context and citation are the real, cited failure entry; the resistance-class "
            "signature is standard biology grounded in the database taxonomy; the hits use the real "
            "fold-change and p-value in your file. It is a hypothesis-generating overlap, not a "
            "diagnosis, and (for a DEG summary) it does not run sample-level causal discovery."
            + ("" if sig_genes else
               " This resistance class is a trial-design or statistical failure with no tumor-intrinsic "
               "gene signature, so only the medical context and your DEG landscape are shown.")),
    }
