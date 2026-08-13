#!/bin/bash
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 backend/server.py  # -> http://localhost:8077
