#!/usr/bin/env bash
set -euo pipefail

echo "Installing Python deps..."
python -m pip install --upgrade pip
pip install -r requirements.txt || true
pip install joblib matplotlib scikit-learn pandas

echo "Generating figures..."
python paper/generate_figs.py

echo "Building LaTeX PDF..."
pushd paper >/dev/null
pdflatex -interaction=nonstopmode main.tex || pdflatex -interaction=nonstopmode main.tex
bibtex main || true
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
popd >/dev/null

echo "Paper build complete: paper/main.pdf"