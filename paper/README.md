Paper build and notes

To build the paper PDF locally (requires a LaTeX distribution such as TeX Live or MikTeX):

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

Alternatively, build the paper using the included Dockerfile (recommended for reproducible builds):

# Build Docker image (from repo root)
docker build -t traindelay-paper -f paper/Dockerfile .

# Run container (it will leave you at a shell); PDF will be at /workspace/paper/main.pdf
docker run --rm -it -v "$(pwd):/workspace" traindelay-paper

Or run the helper script locally (requires pdflatex installed):

./paper/build-paper.sh

Files in this folder:
- `main.tex` — IEEEtran LaTeX file that includes section files.
- `sections/` — individual section files to edit.
- `references.bib` — BibTeX bibliography.
- `figs/` — generated figures (PNG/PDF). The latest figures are: `feature_importance.png`, `pred_vs_true_scatter.png`, `error_coverage.png`, `dataset_stats.png`.

Next steps I can take for you:
- Add a GitHub Actions workflow to build the PDF automatically on push and upload the compiled PDF as an artifact (already added).
- Produce a cover letter for ITSC submission.

I added a Dockerfile and a GitHub Actions workflow (`.github/workflows/build-paper.yml`) that produces `paper/main.pdf` as an artifact. To get a compiled PDF now, either run the Docker build above or push to the repository and let the Action produce the artifact.

If you want I can also generate a short submission checklist and a draft cover letter for ITSC.