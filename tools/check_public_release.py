"""Check Git-selected release files, active links, size and obvious secrets. Stdlib only."""
from pathlib import Path
from urllib.parse import unquote
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                        cwd=ROOT,capture_output=True,check=True)
names = sorted(set(result.stdout.decode('utf-8').split('\0'))-{''})
files = [ROOT/n for n in names if (ROOT/n).is_file()]
errors = []
patterns = {
    'credential URL': re.compile(r'(?i)(?:token|api[_-]?key|access[_-]?token)=[a-z0-9._-]{24,}'),
    'GitHub token': re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    'private key': re.compile(r'^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',re.M),
    'bearer credential': re.compile(r'''(?i)["']authorization["']\s*:\s*["']Bearer\s+[a-z0-9._-]{24,}'''),
}
for p in files:
    rel = p.relative_to(ROOT).as_posix()
    if any(x in p.relative_to(ROOT).parts for x in ('数据','data','.venv','venv')):
        errors.append(f'Private input/environment selected: {rel}')
    if p.is_symlink() or p.stat().st_size>=100_000_000:
        errors.append(f'Unsafe/oversized file: {rel}')
    if p.suffix.lower() not in {'.md','.py','.json','.yml','.yaml','.tex','.csv','.txt','.cff'}:
        continue
    text = p.read_text(encoding='utf-8',errors='replace')
    for label, pattern in patterns.items():
        if pattern.search(text): errors.append(f'{label}: {rel}')
    if p.suffix.lower()=='.md':
        for target in re.findall(r'\]\((<[^>]+>|[^)]+)\)',text):
            target=target.strip('<>').split('#')[0]
            if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target): continue
            if not (p.parent/unquote(target)).exists():
                errors.append(f'Broken local link: {rel} -> {target}')
print(json.dumps({'selected_files':len(files),'selected_bytes':sum(p.stat().st_size for p in files),
                  'status':'PASS' if not errors else 'FAIL','errors':errors},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
