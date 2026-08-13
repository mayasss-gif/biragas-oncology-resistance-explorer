#!/usr/bin/env python3
"""
BiRAGAS causality-engine web app - zero-dependency stdlib server.

Serves the branded frontend and two JSON endpoints backed by the REAL engine:
  GET  /api/example         -> run the built-in causal scenario (cached; instant after warm-up)
  POST /api/run  (CSV body) -> run the engine on an uploaded counts matrix (genes x samples)
"""
import base64
import gzip
import io
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd


def _read_matrix(raw: bytes, filename: str):
    """Parse an uploaded counts matrix (CSV / TSV / TXT / XLSX / XLS, optionally .gz) -> DataFrame."""
    name = (filename or "").lower()
    if name.endswith(".gz") or raw[:2] == b"\x1f\x8b":       # transparently gunzip
        raw = gzip.decompress(raw)
        if name.endswith(".gz"):
            name = name[:-3]                                 # foo.tsv.gz -> foo.tsv (inner format)
    bio = io.BytesIO(raw)
    if name.endswith(".xlsx"):
        return pd.read_excel(bio, index_col=0, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(bio, index_col=0, engine="xlrd")
    if name.endswith(".tsv"):
        return pd.read_csv(bio, index_col=0, sep="\t")
    if name.endswith(".txt"):                                # auto-detect tab / comma / semicolon
        return pd.read_csv(bio, index_col=0, sep=None, engine="python")
    return pd.read_csv(bio, index_col=0)


def _json_clean(o):
    """Recursively replace NaN/Inf (invalid JSON -> breaks the browser's parser) with null."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    return o

HERE = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.join(os.path.dirname(HERE), "frontend")
sys.path.insert(0, HERE)
import engine_api  # noqa: E402
import oncology    # noqa: E402

PORT = int(os.environ.get("BIRAGAS_PORT", "8077"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(_json_clean(body), allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)
        except FileNotFoundError:
            self._send(404, {"error": "not found"})

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self._file(os.path.join(FRONT, "index.html"), "text/html; charset=utf-8")
        elif p.endswith("biragas_logo.png") or p == "/logo":
            self._file(os.path.join(FRONT, "biragas_logo.png"), "image/png")
        elif p == "/api/pathways":
            try:
                self._send(200, oncology.list_pathways())
            except Exception as e:
                self._send(500, {"error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0]
        if p == "/api/run":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            ctype = self.headers.get("Content-Type", "")
            try:
                if "application/json" not in ctype:
                    raise ValueError("expected application/json")
                payload = json.loads(body)
                if not payload.get("data_b64"):
                    raise ValueError("no file uploaded (data_b64 missing)")
                pw_idx = payload.get("pathway_idx")
                if pw_idx is None or pw_idx == "":
                    raise ValueError("choose an oncology resistance pathway first")
                pathway = oncology.get_pathway(pw_idx)
                if pathway is None:
                    raise ValueError(f"unknown pathway index {pw_idx}")
                raw = base64.b64decode(payload["data_b64"])
                filename = payload.get("filename", "upload.csv")
                kind, obj = engine_api.read_upload(raw, filename)          # DEG-table vs matrix
                deg = obj if kind == "deg" else engine_api.matrix_to_deg(obj)
                self._send(200, oncology.run_pathway_analysis(deg, pathway, kind=kind))
            except Exception as e:
                self._send(400, {"error": f"could not parse/run: {e}"})
        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"BiRAGAS oncology resistance app -> http://localhost:{PORT}")
    print(f"loaded {oncology.list_pathways()['n']} oncology resistance pathways")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
