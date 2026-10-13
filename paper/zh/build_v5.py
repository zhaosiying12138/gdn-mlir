#!/usr/bin/env python3
# Build v5 article (41 chapters) -> paper HTML + Zhihu MD twin.
import re, glob, subprocess, os, sys

BASE = os.path.expanduser("~/codebase/gdn-mlir/paper/zh")
PARTS = BASE + "/v5parts"
V4HTML = BASE + "/gdn-mlir-article.html"  # source of the 8 SVG figures
OUT_HTML = BASE + "/gdn-mlir-article.html"
OUT_MD = BASE + "/gdn-mlir-article.md"

FIGDATA = {
 1: "图 1（数据）：前向加速比柱状对比（共载/空闲两窗口）——B1·T2048：2.125× / 1.611×；B4·T2048：1.221× / 1.067×；B1·T8192：1.349× / 0.991×；虚线为 1.0 基准。",
 2: "图 2（数据）：反向阶梯折线（时间对 u）——本方：0.557 / 2.192 / 4.154 / 7.971 / 8.138 / 16.161 ms（u=1/16、1、2、4、4、8）；fla：1.211 / 1.332 / 1.632 / 3.076 / 3.228 / 6.405 ms。终态空闲窗口复测（u=8）：本方 12.15 ms、fla 5.62 ms，比值 0.462。",
 3: "图 3（数据）：新形状调度/调优时间对比，对数刻度——本方冷启动发射 27 µs；本方新形状调度 0.8 s（纯 CPU）；fla 缓存摊销后 0.65–5.67 s；fla 首次搜索 39.8 s（48 个 Triton 配置逐个上机）。",
 4: "图 4（数据）：F1 逐段成本横条（T8192，微秒）——uw 段 116.5；kkt 59.5；solve 41.3；gc 扫描 17.6；phase-0 8.4。",
 5: "图 5（数据）：dkb 位点 dk 输出误差对比（对 fp64 参考，对数刻度）——tf32 单精度路径 1.04e-2（超出容差线 8e-3，被否决）；split-bf16 路径 1.73e-3（容差之内）。split-bf16 乘积残差约 2⁻¹⁷、比 tf32 低约 30 倍，dkb 内核同时从 0.96 提速到 0.25 毫秒。",
 6: "图 6（数据）：反向三大 kernel 静态共享内存池（别名前 → 别名后，门槛 50.2 KB）——dqkwg 100.9 → 47.6 KB；caryA 82.2 → 28.6 KB；wyb 100.1 → 45.1 KB。",
 7: "图 7（数据）：T1 指令族三态判定堆叠条——全部 33 族：8 YES / 19 PARTIAL / 6 NO；feature 层 15 族：8 YES / 6 PARTIAL / 1 NO。",
 8: "图 8（数据）：u=8 的流量下界算术横条（毫秒）——fla Triton（分母）5.62 → 1.00×；本方理想执行（100% 屋顶线）6.1–6.6 → 上限 0.85–0.92×；本方实测 12.15 → 0.462×。",
}

# placeholder number (as written in chapter files after style audit) -> v4 figure index (1-based)
# v4 figure order: 1 forward speedup, 2 bwd ladder, 3 F1 phases, 4 dkb error, 5 arena pools, 6 search cost, 7 T1, 8 ceiling
PLACEHOLDER_TO_V4 = {1: 1, 2: 2, 3: 6, 4: 3, 5: 4, 6: 5, 7: 7, 8: 8}

# 1) collect chapters in numeric order
files = sorted(glob.glob(PARTS + "/CH*.md"), key=lambda f: int(re.search(r"CH(\d+)", os.path.basename(f)).group(1)))
chapters = [open(f, encoding="utf-8").read().strip() for f in files]
master = "\n\n".join(chapters)

# 2) renumber {{FIGn}} by order of appearance -> final numbers (placeholder semantics per PLACEHOLDER_TO_V4)
order = []
def collect(m):
    n = int(m.group(1))
    if n not in order: order.append(n)
    return m.group(0)
re.sub(r"\{\{FIG(\d)\}\}", collect, master)
final_num = {orig: i + 1 for i, orig in enumerate(order)}  # placeholder -> final fig number

# 3) Zhihu MD twin: strip heading hashes, figures -> data lines (keyed by FINAL number)
zh = []
for line in master.split("\n"):
    m = re.match(r"^\{\{FIG(\d)\}\}\s*$", line)
    if m:
        n = final_num[int(m.group(1))]
        zh.append(FIGDATA[n]); zh.append(""); continue
    line = re.sub(r"^### ", "", line)
    line = re.sub(r"^## ", "", line)
    zh.append(line)
zh_text = "# GDN-MLIR：Gated DeltaNet 算子的 MLIR 渐进式下降编译器\n——从语义 op 到 sm_120a cubin 的 MLIR 实践\n\n" + "\n".join(zh)
open(OUT_MD + ".tmp", "w", encoding="utf-8").write(zh_text)
os.replace(OUT_MD + ".tmp", OUT_MD)

# 4) extract the 8 <figure> blocks from the committed v4 html (git HEAD copy), never the live output
v4 = subprocess.run(["git", "-C", os.path.expanduser("~/codebase/gdn-mlir"), "show", "HEAD:paper/zh/gdn-mlir-article.html"],
                    capture_output=True, text=True, check=True).stdout
fig_blocks = re.findall(r"<figure>.*?</figure>", v4, flags=re.S)
assert len(fig_blocks) == 8, f"expected 8 figures in v4, got {len(fig_blocks)}"

# 5) pandoc: master (with h2/h3) -> html body
master_headed = "# GDN-MLIR：Gated DeltaNet 算子的 MLIR 渐进式下降编译器\n\n*从语义 op 到 sm_120a cubin 的 MLIR 实践：编译期调度、PTX 级覆盖证明与交错配对性能裁决。全部代码与 IR 展品取自仓库真实文件，出处随展品标注。*\n\n" + master
open("/tmp/master_v5.md", "w", encoding="utf-8").write(master_headed)
body = subprocess.run(["pandoc", "--from=gfm", "--to=html5", "/tmp/master_v5.md"],
                      capture_output=True, text=True, check=True).stdout

# 6) inject figures: replace <p>{{FIGn}}</p> with the v4 figure block (via PLACEHOLDER_TO_V4), renumbered
for orig, fin in final_num.items():
    blk = fig_blocks[PLACEHOLDER_TO_V4[orig] - 1]
    blk = re.sub(r"图\s*\d+：", f"图 {fin}：", blk, count=1)
    body = body.replace(f'<p>{{{{FIG{orig}}}}}</p>', blk)
assert "{{FIG" not in body, "unreplaced figure placeholder"

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GDN-MLIR：Gated DeltaNet 算子的 MLIR 渐进式下降编译器</title>
<style>
:root{ --ink:#1c2733; --muted:#5b6b7a; --line:#d9e0e7; --accent:#0f5a8a; --soft:#f5f7f9; }
*{box-sizing:border-box}
body{ margin:0; background:#fff; color:var(--ink);
  font-family:"Source Han Serif SC","Noto Serif CJK SC","Songti SC",SimSun,"PingFang SC","Microsoft YaHei",serif;
  font-size:16.5px; line-height:1.9; letter-spacing:.01em; }
main{max-width:900px; margin:0 auto; padding:48px 28px 96px}
h1{font-size:1.62em; line-height:1.5; text-align:center; margin:1.2em 0 .4em; font-weight:700}
h1 + p{ text-align:center; color:var(--muted); margin:.2em 0 1.6em }
h2{font-size:1.28em; margin:2.2em 0 .7em; color:#10222f; font-weight:700; border-bottom:1px solid var(--line); padding-bottom:.3em}
h3{font-size:1.08em; margin:1.6em 0 .5em; color:#17324a; font-weight:700}
p{margin:.7em 0; text-align:justify}
strong{font-weight:700}
table{border-collapse:collapse; width:100%; margin:1em 0; font-size:.93em; line-height:1.6}
th,td{border:1px solid var(--line); padding:.42em .6em; vertical-align:top; text-align:left}
th{background:var(--soft); font-weight:700}
code{font-family:Consolas,"Courier New",monospace; background:#f2f4f6; padding:.08em .35em; border-radius:3px; font-size:.88em}
pre{background:#f2f4f6; border:1px solid var(--line); border-radius:6px; padding:.85em 1em; overflow-x:auto; font-size:.84em; line-height:1.55; margin:1em 0}
pre code{background:none; padding:0}
blockquote{margin:1.1em 0; padding:.6em 1.05em; border-left:4px solid var(--accent); background:var(--soft)}
figure{margin:1.8em 0; text-align:center}
figure svg{max-width:100%; height:auto}
figcaption{font-size:.9em; color:var(--muted); margin-top:.5em; text-align:center; line-height:1.6}
ul,ol{padding-left:1.6em}
li{margin:.3em 0}
.num{font-variant-numeric:tabular-nums}
hr{border:0; border-top:1px solid var(--line); margin:2.4em 0}
</style>
</head>
<body>
<main>
<!--BODY-->
</main>
</body>
</html>
"""
open(OUT_HTML + ".tmp", "w", encoding="utf-8").write(TEMPLATE.replace("<!--BODY-->", body))
os.replace(OUT_HTML + ".tmp", OUT_HTML)

# 7) stats
n_ch = len(chapters)
chars = len(re.sub(r"\s", "", master))
n_code = master.count("```") // 2
n_tab = len(re.findall(r"^\|", master, flags=re.M))
print(f"chapters={n_ch} chars(no-ws)={chars} codeblocks={n_code} tablelines={n_tab}")
print("figure order:", " ".join(f"{o}->{final_num[o]}" for o in order))
print(f"html={os.path.getsize(OUT_HTML)//1024}KB md={os.path.getsize(OUT_MD)//1024}KB")
