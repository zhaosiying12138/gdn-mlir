#!/usr/bin/env python3
"""Build the v6 tutorial from chapter sources into one HTML and one Markdown package."""
from pathlib import Path
import os, re, subprocess, html, hashlib, json

ROOT = Path(__file__).resolve().parents[2]
PARTS = ROOT/'paper/zh/v6parts'
ASSETS = ROOT/'paper/zh/v6assets'
OUT_HTML = ROOT/'paper/zh/gdn-mlir-article.html'
OUT_MD = ROOT/'paper/zh/gdn-mlir-article.md'

TITLE = '从一段 IR 到 GPU 内核：GDN-MLIR 实战教程'
SUBTITLE = '用真实的 Gated DeltaNet 编译器理解 MLIR 方言、pass、结构化 IR、调度、Z3 与 NVVM'

FIG = {
 1: ('fig01-forward-speedup', '前向对 fla Triton 的加速比。共载与空闲窗口分别为每形状 30 对交错计时；绝对时间与两种加速比估计量见第 30 章。'),
 2: ('fig02-backward-ladder', '反向端到端时间随形状负载 u 变化。实线为历史共载窗口的阶梯测量，空心点为终态在另一窗口的复测；历史九 kernel 与当前八 kernel 链分开阅读。'),
 3: ('fig03-search-cost', '新形状调度/调优成本（对数刻度）。本方 0.8 s 为 stage-kernel 的离线求解与排序，fla 39.8 s 为设备首次搜索；完整算子另有实测 A/B。'),
 4: ('fig04-f1-phases', 'F1 在 T8192 下的逐段成本（删除探针，µs）。uw 段占最大份额，分段合计含重叠收益，因此用于判断瓶颈顺序而非机械相加。'),
 5: ('fig05-split-bf16-error', 'dkb 位点的 dk 输出误差（对 fp64 参考，对数刻度）。split-bf16 保留更多尾数并以三条 mma 链接，三链的乘积误差类约为 2^-16；tf32 在该位点超出容差。'),
 6: ('fig06-arena-pools', '生存期别名压缩三个反向 kernel 的静态共享内存池。caryA 对应其退役前的历史构建；每个数字都应与最终后处理后的资源报告一起阅读。'),
 7: ('fig07-t1-coverage', 'T1 指令族覆盖。feature 层按 G6 表逐行重算为 7 YES / 6 PARTIAL / 2 NO；stmatrix 按测量关闭，barrier 的 NO 是命名归并。'),
 8: ('fig08-precision-ceiling', 'u=8 的流量下界算术。理想执行仍需 6.1–6.6 ms，对 fla 5.62 ms 的上限为 0.85–0.92×；实测 12.15 ms 另含执行余量。'),
 9: ('fig09-ir-structure', 'IR 的区域树与 SSA use-def 边。Operation 承载操作数、结果和属性，Region/Block 组织嵌套结构；memref 的 SSA 句柄唯一并不禁止底层内存更新。'),
 10: ('fig10-linalg-kkt-map', 'KKT Gram 的三个 indexing map。共享的 (i,j,r) 分别映射到 k(i,r)、k(j,r) 与输出 (i,j)，第三个迭代维是归约维；掩蔽则由 cmpi 与 select 表达。'),
 11: ('fig11-w32p-bank', 'W32P 的 lane→行映射。80 B 行距等于 20 个字，80 mod 32 = 16，使两个半 warp 落到互补的 16-bank 集合；WST 的 32 位合并存储得以保留。'),
 12: ('fig12-transform-schedule', 'transform 工件通过 handle 选择 payload 中的操作，依次施加多缓冲、异步复制和软件流水。旧 handle 被消费后需要重新 match；解释器执行后再擦除调度。'),
}

PART_NAMES = {
 1: '一、先建立 IR 与方言的心智模型',
 7: '二、把 GDN 数学变成可执行规约',
 9: '三、结构化 IR、内存与转换',
 13: '四、用 IR 编写调度，用 Z3 约束配置',
 17: '五、成本模型与 GPU/NVVM 下降',
 21: '六、机器级实现、验证与迁移',
}
# Part labels used only in the HTML navigation; chapter text itself remains the source of truth.
PART_FOR = {}
for start, name in PART_NAMES.items():
    end = {1:6,7:8,9:12,13:16,17:20,21:32}[start]
    for n in range(start,end+1): PART_FOR[n] = name

files = sorted(PARTS.glob('CH*.md'), key=lambda p:int(re.search(r'CH(\d+)',p.name).group(1)))
assert len(files) == 32, len(files)
chapters = [p.read_text(encoding='utf-8').strip() for p in files]
master = '\n\n'.join(chapters)

# Capture figure appearance order; numbering in the prose is stable by contract.
order=[]
for m in re.finditer(r'\{\{FIG(\d+)\}\}', master):
    n=int(m.group(1))
    assert n in FIG, n
    if n not in order: order.append(n)
assert sorted(order) == list(range(1,13)), order
final_num = {n:i+1 for i,n in enumerate(order)}

# Markdown: retain real headings, use PNG assets and short captions.
zh=[]
for line in master.splitlines():
    m = re.match(r'^\{\{FIG(\d+)\}\}\s*$', line)
    if m:
        old = int(m.group(1)); n=final_num[old]
        asset, caption = FIG[old]
        zh.append(f'![图 {n}：{caption}](v6assets/{asset}.png)')
        zh.append('')
        zh.append(f'*图 {n}：{caption}*')
        zh.append('')
        continue
    zh.append(line)
intro = (
 f'# {TITLE}\n\n'
 f'*{SUBTITLE}*\n\n'
 'GDN-MLIR 以 Gated DeltaNet 前向和反向为实例，把语义方言、可执行算法、调度脚本、约束求解和 GPU 目标代码放在同一条学习路径上。阅读时先弄清 IR 的结构和 pass 的作用，再进入真实内核：每一个抽象都对应仓库中的文件、命令或可复核证据。\n\n'
 '文中将项目实现、上游 MLIR 的通用机制、教程补充的小例和归档测量分开标注。性能数字保留测量窗口、估计量和证据路径；它们用于说明方法的有效范围，不构成对所有形状、所有实现的笼统排名。\n\n'
)
zh_text = intro + '\n'.join(zh) + '\n'
tmp_md = OUT_MD.with_suffix('.md.tmp')
tmp_md.write_text(zh_text, encoding='utf-8')
tmp_md.replace(OUT_MD)

# HTML via pandoc; all chapters and figures come from the same master.
headed = f'# {TITLE}\n\n*{SUBTITLE}*\n\n' + intro.split('\n\n',2)[2] + '\n\n' + master
Path('/tmp/v6_master.md').write_text(headed, encoding='utf-8')
body = subprocess.run(['pandoc','--from=gfm','--to=html5','/tmp/v6_master.md'], check=True, capture_output=True, text=True).stdout
for old,n in final_num.items():
    asset, caption = FIG[old]
    svg = (ASSETS/(asset+'.svg')).read_text(encoding='utf-8')
    # Inline SVGs repeat marker IDs (arr/arrb/arrg). Namespace every id and
    # every url(#...) reference so the composed HTML has no duplicate ids.
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="fig{n}-{m.group(1)}"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#fig{n}-{m.group(1)})', svg)
    block = f'<figure>{svg}<figcaption>图 {n}：{html.escape(caption)}</figcaption></figure>'
    body = body.replace(f'<p>{{{{FIG{old}}}}}</p>', block)
assert '{{FIG' not in body

# Stable chapter ids and a compact navigation, generated from the chapter headings.
ids = iter(range(1,33))
def h2sub(m):
    n = next(ids)
    return f'<h2 id="ch{n:02d}">{m.group(1)}</h2>'
body = re.sub(r'<h2[^>]*>(.*?)</h2>', h2sub, body, flags=re.S)

# Use the original markdown headings for the TOC.
toc=['<nav class="toc"><strong>学习路径</strong>']
current_part = None
for n,p in enumerate(files,1):
    part = PART_FOR[n]
    if part != current_part:
        if current_part is not None: toc.append('</ul>')
        toc.append(f'<p class="toc-part">{html.escape(part)}</p><ul>')
        current_part = part
    heading = chapters[n-1].splitlines()[0]
    heading = re.sub(r'^##\s+', '', heading)
    toc.append(f'<li><a href="#ch{n:02d}">{html.escape(heading)}</a></li>')
toc.append('</ul></nav>')
body = body.replace('<h2 id="ch01">', '\n'.join(toc)+'\n<h2 id="ch01">', 1)

CSS = r'''
:root{--ink:#1c2733;--muted:#5b6b7a;--line:#d9e0e7;--accent:#0f5a8a;--soft:#f5f7f9}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:#fff;color:var(--ink);font-family:"Source Han Serif SC","Noto Serif CJK SC","Songti SC",SimSun,"PingFang SC","Microsoft YaHei",serif;font-size:16.5px;line-height:1.9;letter-spacing:.01em}
main{max-width:960px;margin:0 auto;padding:44px 28px 100px}
h1{font-size:1.65em;line-height:1.45;text-align:center;margin:1.1em 0 .35em;font-weight:700}
h1+p{text-align:center;color:var(--muted);margin:.1em 0 1.4em}
h2{font-size:1.27em;margin:2.25em 0 .7em;color:#10222f;font-weight:700;border-bottom:1px solid var(--line);padding-bottom:.3em;scroll-margin-top:18px}
h3{font-size:1.08em;margin:1.65em 0 .5em;color:#17324a;font-weight:700}
p{margin:.7em 0;text-align:justify}strong{font-weight:700}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.91em;line-height:1.55}
th,td{border:1px solid var(--line);padding:.42em .58em;vertical-align:top;text-align:left}th{background:var(--soft);font-weight:700}
code{font-family:Consolas,"Courier New",monospace;background:#f2f4f6;padding:.08em .32em;border-radius:3px;font-size:.87em}
pre{background:#f2f4f6;border:1px solid var(--line);border-radius:6px;padding:.8em 1em;overflow-x:auto;font-size:.83em;line-height:1.55;margin:1em 0}
pre code{background:none;padding:0}
blockquote{margin:1.1em 0;padding:.6em 1.05em;border-left:4px solid var(--accent);background:var(--soft)}
figure{margin:1.8em 0;text-align:center}figure svg{max-width:100%;height:auto}
figcaption{font-size:.9em;color:var(--muted);margin-top:.5em;text-align:center;line-height:1.6}
ul,ol{padding-left:1.6em}li{margin:.28em 0}
.num{font-variant-numeric:tabular-nums}hr{border:0;border-top:1px solid var(--line);margin:2.4em 0}
.toc{border:1px solid var(--line);background:var(--soft);border-radius:6px;padding:1em 1.25em;margin:1.5em 0 2em;font-size:.92em;line-height:1.55}
.toc>strong{font-size:1.05em}.toc-part{margin:.9em 0 .25em;color:#17324a;font-weight:700}.toc ul{columns:2;column-gap:2em;margin:.25em 0 .4em}.toc li{break-inside:avoid;margin:.18em 0}.toc a{color:var(--accent);text-decoration:none}.toc a:hover{text-decoration:underline}
@media(max-width:650px){main{padding:24px 16px 70px}body{font-size:15.5px}.toc ul{columns:1}table{font-size:.86em}pre{font-size:.78em}}
'''
doc = '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>'+html.escape(TITLE)+'</title>\n<style>'+CSS+'</style>\n</head>\n<body>\n<main>\n'+body+'\n</main>\n</body>\n</html>\n'
tmp_html = OUT_HTML.with_suffix('.html.tmp')
tmp_html.write_text(doc, encoding='utf-8')
tmp_html.replace(OUT_HTML)

# Record exact build inputs and outputs. This distinguishes implementation
# evidence, tutorial-only examples and generated publication artifacts.
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    'format': 'gdn-mlir-article-v6',
    'chapter_count': len(files),
    'figure_count': len(order),
    'source_revision': subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'], capture_output=True, text=True, check=True).stdout.strip(),
    'chapter_sources': {str(p.relative_to(ROOT)): sha(p) for p in files},
    'build_script': sha(Path(__file__)),
    'asset_sources': {str(p.relative_to(ROOT)): sha(p) for p in sorted(ASSETS.glob('*.svg'))},
    'tutorial_examples': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'paper/zh/v6examples').glob('*')) if p.is_file()},
    'outputs': {
        str(OUT_HTML.relative_to(ROOT)): sha(OUT_HTML),
        str(OUT_MD.relative_to(ROOT)): sha(OUT_MD),
    },
    'evidence_note': '代码与 IR 展品来自仓库实现或归档证据；v6examples 中的补充例明确标注为教程练习，不并入项目功能或性能主张。'
}
manifest_path = ROOT/'paper/zh/v6-manifest.json'
tmp_manifest = manifest_path.with_suffix('.json.tmp')
tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
tmp_manifest.replace(manifest_path)

print(f'chapters={len(files)} figures={len(order)} codeblocks={master.count("```")//2} table_lines={len(re.findall(r"^\|",master,re.M))}')
print(f'CJK={len(re.findall("[一-鿿]",master))} html={OUT_HTML.stat().st_size} md={OUT_MD.stat().st_size}')
print('wrote',OUT_HTML,OUT_MD)
