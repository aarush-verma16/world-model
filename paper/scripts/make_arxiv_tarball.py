"""Pack `paper/latex/` into `paper/arxiv_submission.tar.gz` for upload.

Run after compiling with `tectonic -X compile main.tex --keep-intermediates`,
so `main.bbl` exists. arXiv uses an uploaded `.bbl` whose name matches the
main `.tex`, and compiles the source itself with pdflatex.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

LATEX = Path(__file__).resolve().parents[1] / "latex"
OUT = LATEX.parent / "arxiv_submission.tar.gz"


def used_figures() -> set[str]:
    names: set[str] = set()
    for tex in [LATEX / "main.tex", *sorted((LATEX / "sections").glob("*.tex"))]:
        names.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf-8")))
    return names


def main() -> None:
    bbl = LATEX / "main.bbl"
    if not bbl.is_file():
        raise SystemExit("main.bbl missing: compile with --keep-intermediates first")
    files = [LATEX / "main.tex", bbl, LATEX / "refs.bib"]
    files += sorted((LATEX / "sections").glob("*.tex"))
    files += sorted((LATEX / "generated").glob("*.tex"))
    for name in sorted(used_figures()):
        path = LATEX / "figures" / name
        if not path.is_file():
            raise SystemExit(f"figure not found: {path}")
        files.append(path)
    with tarfile.open(OUT, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(LATEX).as_posix())
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(files)} files)")
    for path in files:
        print("  ", path.relative_to(LATEX).as_posix())


if __name__ == "__main__":
    main()
