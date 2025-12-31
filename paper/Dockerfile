FROM python:3.10-slim

# Install TeX Live basics and utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra dvipng make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy repository
COPY . /workspace

RUN python -m pip install --upgrade pip
# Install Python deps; some repos may not need all, but ensure core plotting packages exist
RUN pip install -r requirements.txt || true
RUN pip install joblib matplotlib scikit-learn pandas

WORKDIR /workspace/paper

# Generate figures and build the paper
RUN python generate_figs.py || true
RUN pdflatex -interaction=nonstopmode main.tex || pdflatex -interaction=nonstopmode main.tex
RUN bibtex main || true
RUN pdflatex -interaction=nonstopmode main.tex
RUN pdflatex -interaction=nonstopmode main.tex

CMD ["/bin/bash"]
