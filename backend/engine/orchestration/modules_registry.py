"""
MODULES REGISTRY - the tool table that wires the 24 console modules to the 4 causal layers.

This is the object the supervisor agent dispatches through. Each entry declares the module's tier,
the causal LAYER it belongs to, its role, and whether it can contribute to the causal VERDICT
(only Layers 1-4 can; ingestion/core-signal modules prepare inputs). The registry is the single
source of truth the orchestrator reads when it plans a module chain from a natural-language request.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModuleSpec:
    name: str
    tier: str            # T1 causal, T2 core, T3 process
    layer: str           # ingestion | observational | temporal | genetic | interventional | fusion
    role: str
    verdict_capable: bool   # can this module move an edge toward a CAUSAL verdict?
    method: str


REGISTRY = {
    # ---- L4 interventional (causal verdict) ----
    "causal_calculus": ModuleSpec("Causal calculus", "T1", "interventional",
        "do-calculus identification", True, "back-door/front-door/IV via d-separation"),
    "causal_dag_builder": ModuleSpec("Causal DAG builder", "T1", "observational",
        "structure discovery", True, "bootstrap ensemble PC/FCI/GES/LiNGAM/NOTEARS"),
    "centrality_calculator": ModuleSpec("Centrality calculator", "T1", "interventional",
        "driver ranking", True, "do-effect-weighted centrality"),
    "gwas_mr": ModuleSpec("GWAS / Mendelian randomization", "T1", "genetic",
        "genetic anchoring", True, "two-sample MR + colocalization"),
    "insilico_perturb_patient": ModuleSpec("In silico perturbation (patient)", "T1", "interventional",
        "unit counterfactual", True, "SCM abduction-action-prediction per patient"),
    "insilico_perturb_population": ModuleSpec("In silico perturbation (population)", "T1", "interventional",
        "population counterfactual", True, "Monte-Carlo P(Y|do(T))"),
    "signor_causality": ModuleSpec("SIGNOR causality", "T1", "observational",
        "curated directed prior", True, "literature-anchored orientation prior"),
    # ---- L2 temporal ----
    "temporal_granger": ModuleSpec("Temporal Granger analysis", "T2", "temporal",
        "temporal causality", True, "PCMCI+/conditional Granger"),
    "temporal_bulk": ModuleSpec("Temporal analysis (bulk)", "T2", "temporal",
        "dynamics", False, "differential dynamics + smoothing"),
    # ---- L1 core signal (association, NOT verdict) ----
    "deg_analysis": ModuleSpec("DEG analysis", "T2", "observational",
        "association signal", False, "DESeq2/edgeR/limma-voom"),
    "gsea": ModuleSpec("GSEA pathway enrichment", "T2", "observational",
        "pathway signal", False, "preranked GSEA"),
    "pathway_enrichment": ModuleSpec("Pathway enrichment", "T2", "observational",
        "pathway signal", False, "ORA + topology"),
    "gene_prioritization": ModuleSpec("Gene prioritization", "T2", "observational",
        "candidate triage", False, "score fusion + network priors"),
    "deconvolution": ModuleSpec("Deconvolution", "T2", "observational",
        "cell-type context", False, "reference-based deconvolution"),
    "single_cell": ModuleSpec("Single-cell analysis", "T2", "observational",
        "cellular resolution", False, "QC->embedding->cell typing"),
    "multi_omics": ModuleSpec("Multi-omics analysis", "T2", "observational",
        "layer integration", False, "block fusion + sparse coupling"),
    "total_rna": ModuleSpec("Total RNA pipeline", "T2", "ingestion",
        "primary quant coding+ncRNA", False, "salmon->tximport->DTU->DE->WGCNA->GSEA"),
    # ---- L3/L4 interventional data sources ----
    "perturbation_analysis": ModuleSpec("Perturbation analysis", "T3", "interventional",
        "interventional atlases", True, "DepMap + L1000 differential"),
    "crispr_perturbseq": ModuleSpec("CRISPR Perturb-seq", "T3", "interventional",
        "interventional data", True, "guide-linked scRNA differential"),
    "crispr_screening": ModuleSpec("CRISPR screening", "T3", "interventional",
        "interventional data", True, "pooled screen hit calling"),
    "crispr_amplicon": ModuleSpec("CRISPR targeted / amplicon", "T3", "interventional",
        "interventional validation", True, "on-target edit profiling"),
    # ---- ingestion / process ----
    "cohort_retrieval": ModuleSpec("Cohort retrieval", "T3", "ingestion",
        "data assembly", False, "federated query"),
    "data_harmonization": ModuleSpec("Data harmonization", "T3", "ingestion",
        "batch control", False, "ComBat-style"),
    "fastq_processing": ModuleSpec("FASTQ processing", "T3", "ingestion",
        "primary processing", False, "QC/trim/align/quant"),
}


# canonical causal chain: which modules run, in which layer order, to reach a verdict
CAUSAL_CHAIN = [
    ("ingestion", ["fastq_processing", "total_rna", "cohort_retrieval", "data_harmonization"]),
    ("observational", ["deg_analysis", "pathway_enrichment", "causal_dag_builder", "signor_causality"]),
    ("temporal", ["temporal_granger"]),
    ("genetic", ["gwas_mr"]),
    ("interventional", ["causal_calculus", "insilico_perturb_population",
                        "perturbation_analysis", "centrality_calculator"]),
    ("fusion", []),   # evidence fusion + calibrated CCS + honest gate (engine core)
]


def modules_by_layer(layer: str):
    return {k: v for k, v in REGISTRY.items() if v.layer == layer}


def verdict_capable_modules():
    return {k: v for k, v in REGISTRY.items() if v.verdict_capable}


if __name__ == "__main__":
    print(f"registry: {len(REGISTRY)} modules")
    print(f"verdict-capable (can move an edge toward causal): {len(verdict_capable_modules())}")
    for layer, mods in CAUSAL_CHAIN:
        print(f"  {layer:14} -> {', '.join(mods) if mods else '(engine core: fusion+CCS+gate)'}")
