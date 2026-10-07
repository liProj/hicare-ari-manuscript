"""Build both English PDFs with XeLaTeX. No raw study data are needed."""
from pathlib import Path
import argparse,shutil,subprocess,sys
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--xelatex',default='xelatex');p.add_argument('--regenerate',action='store_true');p.add_argument('--diagrams',action='store_true');a=p.parse_args()
if a.regenerate:subprocess.run([sys.executable,str(R/'scripts/build_manuscript.py')],check=True)
(R/'build').mkdir(exist_ok=True)
def compile_file(name,cwd,out):
    cmd=[a.xelatex,'-interaction=nonstopmode','-halt-on-error',f'-output-directory={out}',name+'.tex']
    for _ in range(2):
        r=subprocess.run(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
        if r.returncode:print(r.stdout[-6000:]);raise SystemExit(r.returncode)
if a.diagrams:
    for name in ['x01_architecture','x34_hicare_detailed']:compile_file(name,R/'figures','.')
for name in ['main','appendix']:
    compile_file(name,R,'build')
    shutil.copy2(R/'build'/f'{name}.pdf',R/f'{name}.pdf')
    print(f'Built {name}.pdf')
