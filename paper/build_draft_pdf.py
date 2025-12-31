from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
import os
import re

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, 'draft_paper.pdf')
SECTIONS_DIR = os.path.join(ROOT, 'sections')
FIGS_DIR = os.path.join(ROOT, 'figs')

# Order of sections to include
section_files = [
    'intro.tex', 'related.tex', 'data.tex', 'methods.tex', 'experiments.tex', 'deployment.tex', 'discussion.tex', 'conclusion.tex'
]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='SectionTitle', fontSize=14, leading=16, spaceAfter=8, spaceBefore=12))
styles.add(ParagraphStyle(name='SubTitle', fontSize=12, leading=14, spaceAfter=6, spaceBefore=8))
styles.add(ParagraphStyle(name='Body', fontSize=10, leading=12))

# Helper to strip simple LaTeX markup
def tex_to_text(tex):
    # Remove LaTeX comments
    tex = re.sub(r"%.*", "", tex)
    # Remove braces around text
    tex = tex.replace('~', ' ')
    tex = re.sub(r"\\(section|subsection)\{([^}]*)\}", r"\n\2\n", tex)
    tex = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", tex)
    tex = re.sub(r"\\item\s*", "- ", tex)
    tex = re.sub(r"\\\[.*?\\\]", "", tex)
    tex = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", tex)
    tex = re.sub(r"\\[a-zA-Z]+", "", tex)
    tex = re.sub(r"\{\s*\\it\s+([^}]*)\s*\}", r"\1", tex)
    # Collapse multiple blank lines
    tex = re.sub(r"\n\s*\n+", "\n\n", tex)
    return tex.strip()

# Read title and authors from main.tex
main_tex = open(os.path.join(ROOT, 'main.tex'), 'r', encoding='utf-8').read()
# Title
m = re.search(r"\\title\{([^}]*)\}", main_tex)
TITLE = m.group(1) if m else 'TrainDelay AI'
# Authors
m = re.search(r"\\author\{([^}]*)\}", main_tex, re.S)
AUTHORS = m.group(1).replace('\\', ', ') if m else ''

# Build document
doc = SimpleDocTemplate(OUT, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
story = []

# Title page
story.append(Paragraph(TITLE, getSampleStyleSheet()['Title']))
story.append(Spacer(1, 6))
story.append(Paragraph(AUTHORS, styles['Body']))
story.append(Spacer(1, 12))
# Abstract from main.tex between begin{abstract} and end{abstract}
m = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", main_tex, re.S)
if m:
    abstract = tex_to_text(m.group(1))
    story.append(Paragraph('<b>Abstract</b>', styles['SubTitle']))
    story.append(Paragraph(abstract, styles['Body']))
    story.append(Spacer(1, 12))

# Keywords
m = re.search(r"\\begin\{IEEEkeywords\}(.+?)\\end\{IEEEkeywords\}", main_tex, re.S)
if m:
    keywords = tex_to_text(m.group(1))
    story.append(Paragraph('<b>Keywords</b>', styles['SubTitle']))
    story.append(Paragraph(keywords, styles['Body']))
    story.append(Spacer(1, 12))

# Add sections
for fname in section_files:
    path = os.path.join(SECTIONS_DIR, fname)
    if not os.path.exists(path):
        continue
    raw = open(path, 'r', encoding='utf-8').read()
    # Extract section title
    title_match = re.search(r"\\section\{([^}]*)\}", raw)
    title = title_match.group(1) if title_match else fname.replace('.tex','')
    story.append(Paragraph(title, styles['SectionTitle']))
    # Remove the section command and convert basic tex
    body = tex_to_text(raw)
    # Remove the section title duplicate if present
    body = re.sub(r"^"+re.escape(title), "", body).strip()
    # Split by paragraphs
    paras = body.split('\n\n')
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # Basic handling for lists
        if p.startswith('- '):
            items = p.split('\n')
            for it in items:
                story.append(Paragraph(it.strip(), styles['Body']))
            story.append(Spacer(1,6))
        else:
            story.append(Paragraph(p.replace('\n',' '), styles['Body']))
            story.append(Spacer(1,6))
    # Insert figures at logical points
    if 'Data' in title:
        fig = os.path.join(FIGS_DIR, 'dataset_stats.png')
        if os.path.exists(fig):
            story.append(Spacer(1,6))
            story.append(Image(fig, width=5*inch, height=1.5*inch))
            story.append(Spacer(1,12))
    if 'Methods' in title:
        fig = os.path.join(FIGS_DIR, 'feature_importance.png')
        if os.path.exists(fig):
            story.append(Spacer(1,6))
            story.append(Image(fig, width=5*inch, height=3*inch))
            story.append(Spacer(1,12))
    if 'Experiments' in title:
        fig1 = os.path.join(FIGS_DIR, 'pred_vs_true_scatter.png')
        fig2 = os.path.join(FIGS_DIR, 'error_coverage.png')
        if os.path.exists(fig1):
            story.append(Spacer(1,6))
            story.append(Image(fig1, width=5*inch, height=4*inch))
            story.append(Spacer(1,12))
        if os.path.exists(fig2):
            story.append(Spacer(1,6))
            story.append(Image(fig2, width=5*inch, height=3*inch))
            story.append(Spacer(1,12))

story.append(PageBreak())
story.append(Paragraph('Acknowledgment', styles['SectionTitle']))
story.append(Paragraph('This draft was generated from repository sources. For IEEE-formatted PDF, compile the LaTeX sources or use the provided Dockerfile.', styles['Body']))

# Build PDF
print('Writing PDF to', OUT)
doc.build(story)
print('Done')
