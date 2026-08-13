"""
INGESTION - raw sequencing to an analysis-ready, provenance-stamped matrix (coding + non-coding).

Modules expanded: FASTQ processing, Total RNA pipeline (Salmon->tximport), Cohort retrieval,
Data harmonization.

This stage does no causal reasoning; its job is to produce honest inputs. Two things matter for the
downstream causal claim:
  * NON-CODING RNA is captured here: Salmon quantifies against the FULL GENCODE transcriptome, so
    lncRNA/miRNA/snoRNA counts come from the same reads as mRNA. If you drop the ncRNA transcripts at
    quantification, no later layer can recover them.
  * BATCH is a CONFOUNDER. Harmonization (ComBat-style) removes platform/batch structure that would
    otherwise be adjusted-for incorrectly (or masquerade as biology) in Layer 4.

Raw inputs are treated as immutable; derived outputs are written to new files with a RunManifest
(seed, genome build, GENCODE release, md5). This reference provides the interfaces and a pure-python
harmonizer; real FASTQ->counts uses nf-core/rnaseq or salmon/STAR (shell-driven), not this file.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RunManifest:
    seed: int = 20260812
    genome_build: str = "GRCh38"
    gencode_release: str = "GENCODE_49"
    accession: str = ""
    input_md5: dict = field(default_factory=dict)
    notes: str = ""

    @staticmethod
    def md5(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


# ------------------------------------------------------------------------------------------
# FASTQ -> counts (INTERFACE ONLY; real work is shell-driven salmon/STAR). Documented so the
# pipeline is complete end-to-end and so the non-coding requirement is explicit.
# ------------------------------------------------------------------------------------------
SALMON_QUANT_CMD = (
    "salmon quant -i {index} -l A -1 {r1} -2 {r2} --gcBias --seqBias "
    "--validateMappings -p {threads} -o {outdir}"
)
# index MUST be built from the GENCODE primary-assembly transcript FASTA that INCLUDES non-coding
# transcripts (lncRNA, miRNA host, snoRNA), so ncRNA is quantified alongside mRNA:
SALMON_INDEX_CMD = "salmon index -t gencode.v49.transcripts.fa.gz -i {index} -k 31 --gencode"


def tximport_note() -> str:
    return ("Aggregate salmon quant.sf to gene/transcript matrices with tximport "
            "(countsFromAbundance='lengthScaledTPM'); keep a transcript-type column so lncRNA/miRNA "
            "nodes are labelled and never silently dropped.")


# ------------------------------------------------------------------------------------------
# Harmonization (batch = confounder). Pure-python location/scale (ComBat-lite) reference.
# ------------------------------------------------------------------------------------------
def harmonize_batches(expr: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    """
    expr: samples x genes (log-scale). batch: sample -> batch id. Removes per-batch location/scale
    shifts (a light ComBat). Real deployment uses sva::ComBat / ComBat-seq with covariate protection.
    """
    out = expr.copy().astype(float)
    grand_mean = out.mean(axis=0)
    grand_sd = out.std(axis=0).replace(0, 1.0)
    for b in batch.unique():
        idx = batch[batch == b].index
        bm = out.loc[idx].mean(axis=0)
        bsd = out.loc[idx].std(axis=0).replace(0, 1.0)
        out.loc[idx] = (out.loc[idx] - bm) / bsd * grand_sd + grand_mean
    return out


def assemble_cohort(count_tables: dict[str, pd.DataFrame], sample_sheet: pd.DataFrame) -> dict:
    """
    count_tables: {study_id -> genes x samples}. sample_sheet: rows (sample, condition, batch, ...).
    Returns a merged samples x genes matrix on the shared gene set, plus the aligned design.
    """
    shared = None
    for t in count_tables.values():
        shared = set(t.index) if shared is None else (shared & set(t.index))
    merged = pd.concat([t.loc[sorted(shared)] for t in count_tables.values()], axis=1)
    merged = merged.T                                       # samples x genes
    design = sample_sheet.set_index("sample").reindex(merged.index)
    return {"expr": merged, "design": design, "n_genes": len(shared), "n_samples": merged.shape[0]}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(8)] + ["MIR29A", "LINC-FIB"]     # note ncRNA nodes present
    s1 = pd.DataFrame(rng.normal(5, 1, (10, 6)), index=genes, columns=[f"s{i}" for i in range(6)])
    s2 = pd.DataFrame(rng.normal(8, 1, (10, 6)), index=genes, columns=[f"t{i}" for i in range(6)])  # batch shift
    sheet = pd.DataFrame({"sample": list(s1.columns) + list(s2.columns),
                          "condition": ["ctrl", "ctrl", "ctrl", "dis", "dis", "dis"] * 2,
                          "batch": ["A"] * 6 + ["B"] * 6})
    coh = assemble_cohort({"study1": s1, "study2": s2}, sheet)
    print(f"assembled {coh['n_samples']} samples x {coh['n_genes']} genes "
          f"(includes ncRNA nodes MIR29A, LINC-FIB)")
    before = coh["expr"].groupby(sheet.set_index("sample")["batch"]).mean().std().mean()
    h = harmonize_batches(coh["expr"], sheet.set_index("sample")["batch"])
    after = h.groupby(sheet.set_index("sample")["batch"]).mean().std().mean()
    print(f"cross-batch spread before harmonize={before:.2f} -> after={after:.2f} (confounder removed)")
