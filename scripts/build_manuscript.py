"""Generate editable English LaTeX from the checked manuscript sources."""
from pathlib import Path
import json,re
R=Path(__file__).resolve().parents[1]
def read(n):return json.loads((R/'source'/n).read_text(encoding='utf8'))
figs={x['n']:x for x in read('figure_manifest.json')}
tables={x['n']:x for x in read('tables.json')}
M=read('main.json'); A=read('appendix.json')
# Inline notation is converted explicitly; displayed equations remain native LaTeX.
maths={'z_i^(m)':r'z_i^{(m)}','z_i^A':r'z_i^A','z_i^B':r'z_i^B','z_i^H':r'z_i^H','p_i^H':r'p_i^H','g_k(i)':r'g_k(i)','q_i':r'q_i','p_i':r'p_i','y_i':r'y_i','x_i':r'x_i','h_i':r'h_i','t_i':r't_i','c_i':r'c_i','v_i':r'v_i','u_i':r'u_i','g_k':r'g_k','O_g':r'O_g','E_g':r'E_g','SMR_g':r'\mathrm{SMR}_g','phi_i':r'\phi_i','theta_t':r'\theta_t','sigma':r'\sigma','tau':r'\tau','10^−7':r'10^{-7}','10^−8':r'10^{-8}'}
def esc(s):
    # Split protected math and URLs before escaping prose.
    pat='|'.join(re.escape(k) for k in sorted(maths,key=len,reverse=True))
    toks=re.split(r'(https?://[^\s]+|'+pat+r')',s)
    out=[]
    for t in toks:
        if t in maths:out.append('$'+maths[t]+'$');continue
        if t.startswith('http'):
            tail='.' if t.endswith('.') else ''
            out.append(r'\url{'+t.rstrip('.')+'}'+tail);continue
        t=t.replace('\\',r'\textbackslash{}')
        t=re.sub(r'([&%$#_{}])',lambda x:'\\'+x[1],t)
        t=t.replace('^',r'\textasciicircum{}').replace('~',r'\textasciitilde{}')
        t=t.replace('≥',r'$\geq$').replace('≤',r'$\leq$').replace('→',r'$\to$').replace('×',r'$\times$').replace('−',r'$-$')
        out.append(t)
    return ''.join(out)
preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec,amsmath,amssymb,graphicx,booktabs,longtable,tabularx,array,geometry,caption,hyperref,needspace,pdflscape}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{Latin Modern Roman}}
\geometry{left=20mm,right=20mm,top=18mm,bottom=19mm}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black,pdftitle={HiCARE: facility profiles and stratified calibration}}
\urlstyle{same}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\setlength{\emergencystretch}{2em}
\setlength{\abovedisplayskip}{7pt}
\setlength{\belowdisplayskip}{7pt}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\newcommand{\mainheading}[1]{\par\needspace{4\baselineskip}\vspace{7pt}{\large\bfseries #1}\par\vspace{3pt}}
\newcommand{\subheading}[1]{\par\needspace{3\baselineskip}\vspace{5pt}{\bfseries #1}\par\vspace{1pt}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\begin{document}
\fontsize{10.5}{13}\selectfont
'''
def title(d):return r'{\raggedright\LARGE\bfseries '+esc(d['title'])+r'\par}\vspace{5pt}'+'\n'+r'{\large '+esc(d['subtitle'])+r'\par}\vspace{10pt}'+'\n'
def fig(b,h=90):
    f=figs[b['n']]; cap=re.sub(r'^Figure (?:S)?\d+\.\s*','',b['caption'])
    return '\n'+r'\begin{center}\includegraphics[width=\linewidth,height='+str(h)+r'mm,keepaspectratio]{figures/'+f['file']+'}\n'+r'\captionof{figure}{'+esc(cap)+'}'+r'\label{'+f['label']+'}\n'+r'\end{center}'+'\n'
def para(s):return '\n\n'.join(esc(p) for p in s.split('\n'))+'\n\n'
def block(b,fh=90):
    typ=b['type']
    if typ=='h1':return '\\mainheading{'+esc(b['text'])+'}\n'
    if typ=='h2':return '\\subheading{'+esc(b['text'])+'}\n'
    if typ=='small':return '{\\small '+esc(b['text'])+'\\par}\n'
    if typ=='p':return para(b['text'])
    if typ=='eq':return '\\begin{equation}\n'+b['latex']+'\n\\end{equation}\n'
    if typ=='exhibit':return fig(b,fh)
    if typ=='ref':return '{\\small ['+str(b['n'])+'] '+esc(b['t'])+' '+esc(b['u'])+'\\par}\n'
    return ''
# Original logical pages are retained to keep text and figures close.
pages=[[]]
for b in M['blocks']:
    if b['type']=='break':pages.append([])
    else:pages[-1].append(b)
# Move the next section opening onto the space freed by deleting the cohort figure.
pages[4].extend(pages[5][:2]); pages[5]=pages[5][2:]
out=[preamble,title(M)]
for pi,p in enumerate(pages):
    nf=sum(b['type']=='exhibit' for b in p)
    words=sum(len((b.get('text') or b.get('caption') or b.get('t') or '').split()) for b in p)
    # Body uses approximately 90 characters per line. Allow space for headings/equations.
    eq=sum(b['type']=='eq' for b in p)
    available=249-words*.155-eq*10-15
    fh=max(45,min(195,available/max(1,nf)-9))
    if pi==8:fh=205
    out.extend(block(b,fh) for b in p)
    if pi<len(pages)-1:out.append('\\clearpage\n')
out.append('\\end{document}\n')
(R/'main.tex').write_text(''.join(out),encoding='utf8')
# Supplementary tables use wrapping columns; all numeric cells are unchanged.
def table(b):
    tb=tables[b['n']]; rows=tb['rows']; n=len(rows[0]); num=b['n']
    cap=re.sub(r'^Table S\d+\.\s*','',b['caption'])
    widths={1:[.19,.52,.29],2:[.32]+[.68/6]*6,3:[.09,.25]+[.66/7]*7,4:[.30]+[.70/9]*9,5:[.34]+[.66/8]*8,6:[.18,.12,.12,.12,.46],7:[.14,.075,.055,.065,.06,.15,.105]+[.35/5]*5,8:[.16,.22]+[.62/7]*7,9:[.31]+[.69/5]*5,10:[.18,.18]+[.64/5]*5,11:[.18]+[.82/6]*6,12:[.10,.24]+[.66/5]*5,13:[.12,.23]+[.65/5]*5,14:[.45]+[.55/6]*6}[num]
    # Subtract column padding from total width before assigning fractions.
    landscape=num in [4,7]
    pagewidth=260 if landscape else 170
    total=pagewidth-(n-1)*2.2-.2
    col='@{}'+''.join('L{'+f'{total*w:.2f}'+'mm}' for w in widths)+'@{}'
    z=([r'\clearpage'] if num!=1 else [])+([r'\begin{landscape}'] if landscape else [])+[r'\begingroup\fontsize{8}{9.5}\selectfont\setlength{\tabcolsep}{1.1mm}\renewcommand{\arraystretch}{1.2}',r'\begin{longtable}{'+col+'}',r'\caption{'+esc(cap)+'}'+r'\label{'+tb['label']+r'}\\',r'\toprule']
    def row(r):return ' & '.join(esc(c) for c in r)+r' \\'
    if num==7:z=[s.replace(r'\arraystretch}{1.2}',r'\arraystretch}{1.0}') for s in z]
    z += [row(rows[0]),r'\midrule\endfirsthead',r'\multicolumn{'+str(n)+r'}{l}{\small Table S'+str(num)+r' continued}\\',r'\toprule',row(rows[0]),r'\midrule\endhead',r'\bottomrule\endfoot']
    for i,r in enumerate(rows[1:],1):
        if i in tb['spans']:z.append(r'\multicolumn{'+str(n)+r'}{p{167mm}}{\bfseries '+esc(r[0])+r'}\\');continue
        z.append(row(r))
    z += [r'\end{longtable}\endgroup']
    if b.get('note'):z.append('{\\small '+esc(b['note'])+'\\par}')
    z.append(para(b['analysis']))
    if landscape:z.append(r'\end{landscape}')
    return '\n'.join(z)+'\n'
out=[preamble,r'\renewcommand{\thefigure}{S\arabic{figure}}',r'\renewcommand{\thetable}{S\arabic{table}}',r'\renewcommand{\theequation}{S\arabic{equation}}',title(A)]
for b in A['blocks']:
    if b['type']=='exhibit' and b['kind']=='table':out.append(table(b));continue
    if b['type']=='exhibit':
        out.extend(([] if b['n']==3 else ['\\clearpage\n'])+[fig(b,145),para(b['analysis'])])
        if b.get('note'):out.append('{\\small '+esc(b['note'])+'\\par}\n')
        continue
    if b['type']=='h1' and b['text'].startswith(('S2 ','S3 ','S4 ')):out.append('\\clearpage\n')
    out.append(block(b))
out.append('\\end{document}\n')
(R/'appendix.tex').write_text(''.join(out),encoding='utf8')
print('Generated main.tex and appendix.tex')
