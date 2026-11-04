"""Rebuild all figures and compile the report with a preinstalled XeLaTeX."""
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
engine = shutil.which('xelatex')
if engine is None:
    raise SystemExit('Install XeLaTeX and put it on PATH; see README.md for fonts/packages.')
subprocess.run([sys.executable, str(root/'build_figures.py')], cwd=root, check=True)
(root/'build').mkdir(exist_ok=True)
for _ in range(2):
    subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error',
                    '-no-shell-escape', '-output-directory=build', 'main.tex'],
                   cwd=root, check=True)
log = (root/'build/main.log').read_text(encoding='utf-8', errors='replace')
for problem in ['Missing character:', 'undefined references', 'Overfull \\hbox', 'Overfull \\vbox']:
    if problem in log:
        raise SystemExit(f'Layout check failed: {problem}; inspect build/main.log')
shutil.copy2(root/'build/main.pdf', root/'report.pdf')
print(f'Written: {root / "report.pdf"}')
