#!/usr/bin/env python3
"""
Live public-data enrichment for BiRAGAS targets - the same sources the popular MCP servers wrap, queried
directly from the backend (no key required). For a gene symbol we pull, best-effort and cached:

  Ensembl   REST   -> canonical gene id, biotype, chromosome, description
  UniProt   REST   -> reviewed accession (for AlphaFold)
  AlphaFold API    -> predicted 3D structure availability + model URL (structure-based design feasible?)
  ChEMBL    REST   -> is it a drug target? existing drugs, max clinical phase, mechanism of action
  PubChem   PUG    -> CID for the top known drug (compound link)
  GTEx v2   API    -> gencodeId + highest-median-expression tissue
  cBioPortal REST  -> Entrez id + a cross-cancer mutation query link

Every call is wrapped with a short timeout and degrades to a partial result; nothing is fabricated.
Reusable by all three BiRAGAS apps.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

UA = "BiRAGAS-live-data/1.0 (research; mayass@ayassbioscience.com)"
_CACHE: dict = {}


def _get(url, timeout=12, headers=None):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json",
                                                   **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def ensembl_gene(symbol):
    d = _get(f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{urllib.parse.quote(symbol)}"
             "?content-type=application/json")
    if not d:
        return None
    return {"ensembl_id": d.get("id"), "biotype": d.get("biotype"),
            "chromosome": d.get("seq_region_name"),
            "description": (d.get("description") or "").split(" [")[0]}


def uniprot_accession(symbol):
    d = _get("https://rest.uniprot.org/uniprotkb/search?query="
             f"gene_exact:{urllib.parse.quote(symbol)}+AND+organism_id:9606+AND+reviewed:true"
             "&fields=accession&format=json&size=1")
    r = (d or {}).get("results") or []
    return r[0]["primaryAccession"] if r else None


def alphafold(accession):
    if not accession:
        return {"available": False}
    d = _get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")
    if d and isinstance(d, list) and d:
        return {"available": True, "uniprot": accession,
                "model_url": d[0].get("pdbUrl") or d[0].get("cifUrl"),
                "viewer": f"https://alphafold.ebi.ac.uk/entry/{accession}"}
    return {"available": False, "uniprot": accession}


def chembl_drugs(symbol, accession=None):
    tid = None
    if accession:                                        # precise: map by UniProt accession (fixes AR, etc.)
        td = _get("https://www.ebi.ac.uk/chembl/api/data/target?target_components__accession="
                  f"{accession}&format=json")
        tgs = [t for t in (td or {}).get("targets", [])
               if t.get("target_type") == "SINGLE PROTEIN" and t.get("organism") == "Homo sapiens"]
        if tgs:
            tid = tgs[0]["target_chembl_id"]
    if not tid:                                          # fallback: name search
        ts = _get(f"https://www.ebi.ac.uk/chembl/api/data/target/search?q={urllib.parse.quote(symbol)}&format=json")
        targets = [t for t in (ts or {}).get("targets", [])
                   if t.get("organism") == "Homo sapiens"
                   and (t.get("pref_name", "").upper() == symbol.upper() or t.get("target_type") == "SINGLE PROTEIN")]
        targets = targets or [t for t in (ts or {}).get("targets", []) if t.get("organism") == "Homo sapiens"]
        if not targets:
            return {"is_target": False}
        tid = targets[0]["target_chembl_id"]
    mech = _get(f"https://www.ebi.ac.uk/chembl/api/data/mechanism?target_chembl_id={tid}&format=json&limit=20")
    mechs = (mech or {}).get("mechanisms", [])
    if not mechs:
        return {"is_target": True, "target_chembl_id": tid, "n_drugs": 0}
    ids = [m["molecule_chembl_id"] for m in mechs if m.get("molecule_chembl_id")]
    actions = {m["molecule_chembl_id"]: m.get("action_type") or m.get("mechanism_of_action")
               for m in mechs if m.get("molecule_chembl_id")}
    mol = _get("https://www.ebi.ac.uk/chembl/api/data/molecule?molecule_chembl_id__in="
               + ",".join(ids[:15]) + "&format=json&limit=15")
    def _phase(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    drugs = []
    for m in (mol or {}).get("molecules", []):
        drugs.append({"name": m.get("pref_name") or m.get("molecule_chembl_id"),
                      "max_phase": _phase(m.get("max_phase")),
                      "action": actions.get(m.get("molecule_chembl_id"))})
    drugs.sort(key=lambda d: (d["max_phase"] if d["max_phase"] is not None else -1), reverse=True)
    max_phase = max([d["max_phase"] for d in drugs if d["max_phase"] is not None] or [0])
    return {"is_target": True, "target_chembl_id": tid, "n_drugs": len(set(ids)),
            "max_phase": max_phase, "drugs": drugs[:6]}


def pubchem_cid(drug_name):
    if not drug_name:
        return None
    d = _get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
             f"{urllib.parse.quote(drug_name)}/cids/JSON", timeout=10)
    cids = (((d or {}).get("IdentifierList") or {}).get("CID") or [])
    return cids[0] if cids else None


def gtex_expression(symbol):
    g = _get(f"https://gtexportal.org/api/v2/reference/gene?geneId={urllib.parse.quote(symbol)}")
    rows = (g or {}).get("data") or []
    if not rows:
        return None
    gencode = rows[0]["gencodeId"]
    e = _get(f"https://gtexportal.org/api/v2/expression/medianGeneExpression?gencodeId={gencode}"
             "&datasetId=gtex_v8")
    data = (e or {}).get("data") or []
    top = max(data, key=lambda x: x.get("median", 0)) if data else None
    return {"gencodeId": gencode,
            "top_tissue": (top.get("tissueSiteDetailId") if top else None),
            "top_median_tpm": (round(top.get("median"), 2) if top else None)}


def cbioportal(symbol):
    d = _get(f"https://www.cbioportal.org/api/genes/{urllib.parse.quote(symbol)}")
    if not d or not d.get("entrezGeneId"):
        return None
    return {"entrez": d["entrezGeneId"],
            "url": f"https://www.cbioportal.org/results?gene_list={urllib.parse.quote(symbol)}"
                   "&cancer_study_list=5c8a7d55e4b046111fee2296"}   # cross-cancer (TCGA PanCancer Atlas)


def enrich_target(symbol):
    """Aggregate all live sources for one gene symbol (cached)."""
    if symbol in _CACHE:
        return _CACHE[symbol]
    ens = ensembl_gene(symbol)
    acc = uniprot_accession(symbol)
    chembl = chembl_drugs(symbol, accession=acc)
    top_drug = (chembl.get("drugs") or [{}])[0].get("name") if chembl.get("drugs") else None
    out = {
        "gene": symbol,
        "ensembl": ens,
        "structure": alphafold(acc),
        "chembl": chembl,
        "gtex": gtex_expression(symbol),
        "cbioportal": cbioportal(symbol),
        "pubchem_cid": pubchem_cid(top_drug),
        "links": {
            "opentargets": (f"https://platform.opentargets.org/target/{ens['ensembl_id']}"
                            if ens and ens.get("ensembl_id") else None),
            "genecards": f"https://www.genecards.org/cgi-bin/carddisp.pl?gene={urllib.parse.quote(symbol)}",
            "pubchem": (f"https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_cid(top_drug)}"
                        if top_drug else None),
        },
    }
    _CACHE[symbol] = out
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(enrich_target(sys.argv[1] if len(sys.argv) > 1 else "FLT3"), indent=2))
