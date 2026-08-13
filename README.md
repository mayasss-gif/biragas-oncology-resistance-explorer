# BiRAGAS Oncology Resistance Explorer

Pick one of **209 real, cited oncology therapy-resistance pathways**, upload a differential-expression file (DEG table or genes x samples matrix), and map the sample onto that resistance mechanism: real medical context, the resistance-class gene signature altered in the sample, a volcano, and **live target intelligence** (Ensembl, ChEMBL existing drugs, AlphaFold structure, GTEx, cBioPortal, PubChem) with a rational-replacement drug line.

## Quick start
```bash
pip install -r requirements.txt
python3 backend/server.py            # or ./run.command
# open http://localhost:8077
```
Zero web framework: a stdlib `http.server` backend + a self-contained HTML/JS frontend (Plotly via CDN).
Live lookups (OpenTargets, Ensembl, ChEMBL, AlphaFold, GTEx, cBioPortal, PubChem) are keyless public APIs,
so the browser / server needs internet access.

## Layout
```
backend/
  server.py        stdlib HTTP server + JSON API
  engine/          bundled four-layer causal engine (layers + orchestration)
  data/            public reference data (GTEx demo matrices, cited pathways, class signatures)
frontend/
  index.html       self-contained UI
```

## Data & honesty
The 209-pathway dataset is compiled from cited literature (PMIDs / ClinicalTrials.gov NCTs). Nothing is fabricated: results use the real values in your uploaded file plus real public
data. **No patient data is included in this repository** (patient DEG files are git-ignored).

## About
Part of the BiRAGAS platform by **AYASS BIO-SCIENCE, LLC**. Contact: mayass@ayassbioscience.com
