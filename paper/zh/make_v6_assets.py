#!/usr/bin/env python3
import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'paper/zh/gdn-mlir-article.html'
OUT = ROOT/'paper/zh/v6assets'
OUT.mkdir(parents=True, exist_ok=True)

names = {
 1:'fig01-forward-speedup', 2:'fig02-backward-ladder', 3:'fig03-search-cost',
 4:'fig04-f1-phases', 5:'fig05-split-bf16-error', 6:'fig06-arena-pools',
 7:'fig07-t1-coverage', 8:'fig08-precision-ceiling',
 9:'fig09-ir-structure', 10:'fig10-linalg-kkt-map',
 11:'fig11-w32p-bank', 12:'fig12-transform-schedule'
}

# Bind archived figures by their visible titles, never by chapter order alone.
# Keep the archived eight-figure article revision fixed so rebuilding after the
# v6 article has more figures cannot accidentally ingest the new document.
BASE_ARTICLE_REV = '9ab68e7'
old = subprocess.run(['git','-C',str(ROOT),'show',BASE_ARTICLE_REV+':paper/zh/gdn-mlir-article.html'], check=True, capture_output=True, text=True).stdout
svgs = re.findall(r'<svg\b.*?</svg>', old, flags=re.S)
assert len(svgs) == 8, len(svgs)
for svg in svgs:
    title = re.search(r'<text[^>]*>([^<]+)', svg).group(1)
    if '前向加速比' in title: i = 1
    elif '反向端到端时间对形状负载' in title: i = 2
    elif '新形状的调度/调优时间' in title: i = 3
    elif 'F1 逐段成本' in title: i = 4
    elif 'dkb 位点 dk 输出误差' in title: i = 5
    elif '反向三大 kernel 的静态共享内存池' in title: i = 6
    elif 'T1 指令族三态判定' in title: i = 7
    elif '流量下界算术' in title: i = 8
    else: raise ValueError(title)
    if i == 7:
        continue  # regenerated below with a row-by-row corrected feature count
    (OUT/(names[i]+'.svg')).write_text(svg, encoding='utf-8')

FONT = 'Noto Sans CJK SC, Microsoft YaHei, PingFang SC, sans-serif'
MONO = 'Consolas, Courier New, monospace'

def esc(s):
    return (s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            .replace('"','&quot;'))

def text(x,y,s,size=14,fill='#1c2733',weight='normal',anchor='start',family=FONT):
    # Cairo/Pango fallback is unreliable for CJK or superscript glyphs inside
    # a Western monospace span. Use the CJK font for mixed/non-ASCII text.
    if any(ord(ch) > 127 for ch in s):
        family = FONT
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'

def rect(x,y,w,h,fill='#f5f7f9',stroke='#9db2c2',rx=6,sw=1.2,dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def line(x1,y1,x2,y2,color='#5b6b7a',dash=None,marker=True,sw=1.5):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = ' marker-end="url(#arr)"' if marker else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>'

def svgwrap(w,h,body,defs=True):
    d = '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#5b6b7a"/></marker></defs>' if defs else ''
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img">{d}{body}</svg>'

# FIG 7: authoritative row-by-row recount of the G6 feature table.
b=[]
b.append(text(30,28,'T1 指令族覆盖：总量与 feature 层分开计数',16,weight='bold'))
b.append(text(30,64,'全部 33 族',14))
for x,w,c,label in [(150,127,'#2e7d5b','8 YES'),(277,302,'#b8860b','19 PARTIAL'),(579,96,'#8a8f94','6 NO')]:
    b.append(rect(x,52,w,24,c,stroke='none',rx=0,sw=0)); b.append(text(x+w/2,70,label,12,fill='#fff',anchor='middle'))
b.append(text(30,118,'feature 层 15 族（逐行重算）',14))
for x,w,c,label in [(150,244,'#2e7d5b','7 YES'),(394,209,'#b8860b','6 PARTIAL'),(603,70,'#8a8f94','2 NO')]:
    b.append(rect(x,102,w,24,c,stroke='none',rx=0,sw=0)); b.append(text(x+w/2,120,label,12,fill='#fff',anchor='middle'))
b.append(text(30,158,'feature 层的两个 NO：stmatrix 按测量关闭；barrier 是命名归并，均不表示能力缺口。',12,fill='#5b6b7a'))
(OUT/(names[7]+'.svg')).write_text(svgwrap(740,185,''.join(b)),encoding='utf-8')

# FIG 9: IR containment tree vs use-def edges.
b=[]
b.append(text(24,26,'IR 的两套结构：区域树与 SSA 的 use-def 边',16,weight='bold'))
b.append(rect(20,48,385,310,'#f7fafc'))
b.append(text(34,72,'真实 IR（节选）',13,weight='bold'))
code=['func.func @smoke(%q, %k, %v, ...) {',
      '  %o, %ht = "gdn.fwd"(%q, %k, %v, ...)',
      '      <{config = #gdn.config<64, ...>}>',
      '  return %o, %ht',
      '}',
      '',
      'linalg.generic {',
      '  ^bb0(%a, %b, %acc):',
      '    %p = arith.mulf %a, %b',
      '    %s = arith.addf %acc, %p',
      '    linalg.yield %s',
      '}']
for i,s in enumerate(code): b.append(text(34,99+i*22,s,11,family=MONO))
b.append(rect(450,48,420,310,'#ffffff'))
b.append(text(465,72,'包含关系',13,weight='bold'))
b.append(rect(475,88,170,36,'#eef4f9')); b.append(text(490,111,'module / func region',12))
b.append(rect(495,140,190,36,'#eef4f9')); b.append(text(510,163,'block ^bb0',12))
b.append(rect(515,192,190,36,'#eef4f9')); b.append(text(530,215,'gdn.fwd（Operation）',12))
b.append(line(560,124,560,140,marker=False)); b.append(line(580,176,580,192,marker=False))
b.append(text(470,260,'use-def（另一套边）',13,weight='bold'))
b.append(text(480,286,'%q  ───────────────▶  gdn.fwd',12,family=MONO))
b.append(text(480,312,'%o   ───────────────▶  return',12,family=MONO))
b.append(text(480,338,'%p   ───────────────▶  arith.addf',12,family=MONO))
b.append(text(24,390,'操作数/结果/属性属于 Operation；region 包含 block，block 包含操作。SSA 的定义—使用关系沿另一套边连接。',12,fill='#5b6b7a'))
b.append(text(24,414,'memref 的 SSA 句柄唯一，不代表其指向的内存不可写；别名与副作用需要另行分析。',12,fill='#5b6b7a'))
(OUT/(names[9]+'.svg')).write_text(svgwrap(900,440,''.join(b)),encoding='utf-8')

# FIG 10: KKT indexing maps with coordinates and a mask.
b=[]
b.append(text(24,26,'linalg.generic 如何共享迭代坐标：KKT 的三个 map',16,weight='bold'))
b.append(rect(20,48,380,250,'#f7fafc'))
b.append(text(34,73,'迭代域 (i, j, r)',14,weight='bold'))
for i,s in enumerate(['A: (i,j,r) → (i,r)', 'B: (i,j,r) → (j,r)', 'O: (i,j,r) → (i,j)']):
    b.append(text(40,108+i*32,s,14,family=MONO))
b.append(text(34,225,'iterator_types =',12,fill='#5b6b7a'))
b.append(text(34,250,'[parallel, parallel, reduction]',12,family=MONO))
b.append(text(34,278,'r 是归约维；求和由 region 的',12,fill='#5b6b7a'))
b.append(text(34,298,'addf + yield 显式完成。',12,fill='#5b6b7a'))
b.append(rect(440,48,430,250,'#ffffff'))
b.append(text(455,73,'KKT Gram 的实际映射',14,weight='bold'))
for i,s in enumerate(['A[i,r] = k[i,r]', 'B[j,r] = k[j,r]', 'O[i,j] += A[i,r] · B[j,r]']):
    b.append(text(465,108+i*32,s,14,family=MONO))
b.append(text(455,225,'因此 O[i,j] = Σ_r k[i,r]·k[j,r]',14,weight='bold'))
b.append(text(455,252,'即 k·k^T，而不是教科书式 B[k,n]。',12,fill='#5b6b7a'))
b.append(rect(20,322,850,78,'#fff7e6',stroke='#d9b66f'))
b.append(text(34,348,'另一个 generic 用下标表达掩蔽：A[i,j] = select(j < i, value, 0)',13,family=MONO))
b.append(text(34,376,'strict_tril 由 cmpi slt + select 完成；P 把谓词换成 sle，包含对角。',12,fill='#5b6b7a'))
(OUT/(names[10]+'.svg')).write_text(svgwrap(900,425,''.join(b)),encoding='utf-8')

# FIG 11: bank complement map with real address intervals.
b=[]
b.append(text(24,26,'W32P：让两个半 warp 落在互补的 16 个 bank 上',16,weight='bold'))
b.append(text(28,56,'row = 8·(i >> 7) + warp + 4·(lane ≥ 16)      col = 2·(lane mod 16)',13,family=MONO))
b.append(text(28,84,'F2 tile：行距 80 B = 20 个字；bank = 字地址 mod 32；80 mod 32 = 16',13,fill='#5b6b7a'))
# two address rows
b.append(text(28,124,'lane 0–15：行 r',13,weight='bold',fill='#0f5a8a'))
b.append(text(28,151,'地址 [20r, 20r+15]',12,family=MONO))
b.append(text(28,184,'lane 16–31：行 r+4',13,weight='bold',fill='#b06020'))
b.append(text(28,211,'地址 [20r+80, 20r+95]',12,family=MONO))
# bank grid
x0,y0,cell=310,118,25
for bank in range(32):
    x=x0+(bank%16)*cell; y=y0+(bank//16)*52
    color='#8bbbd6' if bank<16 else '#e5b38c'
    b.append(rect(x,y,cell-2,42,color,stroke='#ffffff',rx=2,sw=1))
    b.append(text(x+(cell-2)/2,y+26,str(bank),11,anchor='middle'))
b.append(text(310,254,'bank 0–15',12,fill='#0f5a8a'))
b.append(text(710,254,'bank 16–31',12,fill='#b06020',anchor='end'))
# arrows from rows to grid labels
b.append(line(215,151,292,151,color='#0f5a8a'))
b.append(line(215,211,292,211,color='#b06020'))
b.append(text(28,302,'低半 warp 覆盖 bank 0–15；高半 warp 的 80 字偏移等价于 bank 编号整体平移 16。',13))
b.append(text(28,330,'两组访问合起来覆盖 32 个 bank，冲突从 2 路降为 0；WST 仍保留 32 位合并存储。',13))
b.append(text(28,362,'结果：七场景 −17.9 至 −22.6 µs，池与驻留不变；下标变换只改变 lane 与行的双射。',12,fill='#5b6b7a'))
(OUT/(names[11]+'.svg')).write_text(svgwrap(900,395,''.join(b)),encoding='utf-8')

# FIG 12: actual transform schedule handles and resulting IR.
b=[]
b.append(text(24,26,'transform：句柄选择目标，工件只改变调度',16,weight='bold'))
b.append(rect(20,52,250,250,'#f7fafc'))
b.append(text(34,78,'payload（算法 IR）',13,weight='bold'))
for i,s in enumerate(['memref.alloc', 'scf.for', '  cp.async / mma', '  linalg.matmul']):
    b.append(text(38,111+i*35,s,13,family=MONO))
b.append(rect(315,52,280,250,'#eef4f9'))
b.append(text(329,78,'transform 工件',13,weight='bold'))
for i,s in enumerate(['match memref.alloc', 'multibuffer factor=2', 'create_async_groups', 'match scf.for', 'pipeline depth=2']):
    b.append(text(329,111+i*32,s,11,family=MONO))
b.append(rect(640,52,230,250,'#f0f7f0'))
b.append(text(654,78,'调度后的 payload',13,weight='bold'))
for i,s in enumerate(['A/B 双缓冲', '异步复制组', 'K 循环 depth=2', 'peel_epilogue']):
    b.append(text(654,111+i*35,s,12,family=MONO))
b.append(line(270,175,315,175)); b.append(line(595,175,640,175))
b.append(text(276,162,'handle',11,fill='#5b6b7a'))
b.append(text(598,162,'apply',11,fill='#5b6b7a'))
b.append(rect(20,328,850,52,'#fff7e6',stroke='#d9b66f'))
b.append(text(34,360,'合并 → transform-interpreter → erase-schedule → gpu-lower-to-nvvm-pipeline',12,family=MONO))
b.append(text(24,408,'改写会消费旧 handle；后续用新 handle 或重新 match。调度文本与数学 payload 分开保存。',12,fill='#5b6b7a'))
(OUT/(names[12]+'.svg')).write_text(svgwrap(900,435,''.join(b)),encoding='utf-8')

# Convert all SVGs to PNG for the markdown package. HTML will keep the SVG inline.
sys.path.insert(0, '/tmp/v6svg')
import cairosvg
for i,n in names.items():
    src = OUT/(n+'.svg')
    cairosvg.svg2png(url=str(src), write_to=str(OUT/(n+'.png')), output_width=1400)
print('assets:', len(list(OUT.glob('*.svg'))), 'svg,', len(list(OUT.glob('*.png'))), 'png')
