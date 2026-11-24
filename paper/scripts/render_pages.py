"""Render `paper/latex/main.pdf` to PNGs in `paper/latex/build_preview/` for a visual check."""

from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1] / "latex"
OUT = ROOT / "build_preview"
OUT.mkdir(exist_ok=True)
doc = fitz.open(ROOT / "main.pdf")
print(f"{doc.page_count} pages")
for i, page in enumerate(doc):
    page.get_pixmap(dpi=80).save(OUT / f"page_{i + 1:02d}.png")
