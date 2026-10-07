"""Check figure/table completeness, English text and compiled manuscript PDFs."""
from pathlib import Path
import json,re,ast
import pymupdf
R=Path(__file__).resolve().parents[1]
def load(n):return json.loads((R/'source'/n).read_text(encoding='utf8'))
m=load('main.json');a=load('appendix.json');tables=load('tables.json');figs=load('figure_manifest.json')
assert len(figs)==33 and not any(f['n']==1 for f in figs)
assert len(tables)==14
assert sum(b['type']=='exhibit' and b.get('kind')=='figure' for b in m['blocks'])==27
assert sum(b['type']=='exhibit' and b.get('kind')=='figure' for b in a['blocks'])==6
assert sum(b['type']=='ref' for b in m['blocks'])==35
assert not any('cohort_flow' in f['file'] for f in figs)
for f in figs:
 p=R/'figures'/f['file'];assert p.exists(),p
 text=''.join(page.get_text() for page in pymupdf.open(p))
 assert not re.search(r'[\u4e00-\u9fff]',text),p
for n in ['main.tex','appendix.tex']:
 text=(R/n).read_text(encoding='utf8');assert not re.search(r'[\u4e00-\u9fff]',text),n
 assert 'fig:flow' not in text and 'fig01_cohort_flow' not in text
for p in (R/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
report={'main_figures':27,'supplementary_figures':6,'tables':14,'references':35,'analysis_scripts':len(list((R/'code').glob('*.py')))}
for n in ['main','appendix']:
 d=pymupdf.open(R/f'{n}.pdf');report[n+'_pages']=len(d)
 text=''.join(p.get_text() for p in d)
 assert not re.search(r'[\u4e00-\u9fff]',text),n
 assert all(len(p.get_text().strip())>200 for p in d),'nearly empty page in '+n
 log=(R/'build'/f'{n}.log')
 if log.exists():
  log=log.read_text(encoding='utf8',errors='replace')
  assert not re.search(r'Overfull|Missing character:|undefined references',log),n
assert report['main_pages']==30
(R/'build'/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
