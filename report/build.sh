#!/bin/sh
# Rebuild the report from the cached results (report/results/*.json; rerun report/mkresults.py to refresh them) and copy the PDF to the
# repository root.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY mktables.py
$PY figures.py
tectonic -X compile pywopwop_report.tex
cp pywopwop_report.pdf ../PyWopWop-report.pdf
