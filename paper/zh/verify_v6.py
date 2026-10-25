#!/usr/bin/env python3
from pathlib import Path
from html.parser import HTMLParser
import re, os, json, hashlib
ROOT=Path(__file__).resolve().parents[2]
md=ROOT/'paper/zh/gdn-mlir-article.md'
htmlf=ROOT/'paper/zh/gdn-mlir-article.html'
parts=ROOT/'paper/zh/v6parts'
assets=ROOT/'paper/zh/v6assets'
text=md.read_text(encoding='utf-8')
h=htmlf.read_text(encoding='utf-8')
class P(HTMLParser):
    def __init__(self): super().__init__(); self.ids=[]; self.hrefs=[]; self.figures=0; self.tables=0
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d: self.ids.append(d['id'])
        if tag=='a' and 'href' in d: self.hrefs.append(d['href'])
        if tag=='figure': self.figures+=1
        if tag=='table': self.tables+=1
p=P(); p.feed(h)
chapters=[f'ch{i:02d}' for i in range(1,33)]
assert all(x in p.ids for x in chapters), [x for x in chapters if x not in p.ids]
assert len(set(p.ids))==len(p.ids)
assert p.figures==12,p.figures
for href in p.hrefs:
    if href.startswith('#'): assert href[1:] in p.ids, href
assert '{{FIG' not in h and '{{FIG' not in text
assert len(re.findall(r'^## ',text,re.M))==32
assert len(re.findall(r'^### \d+\. ',text,re.M))>=150
imgs=re.findall(r'!\[[^]]*\]\(([^)]+)\)',text)
assert len(imgs)==12,len(imgs)
for src in imgs:
    assert (ROOT/'paper/zh'/src).exists(),src
# headings and image captions
assert text.startswith('# 从一段 IR 到 GPU 内核')
for phrase in ['feature 层 15 族按 G6 表逐行重算为 7 YES、6 PARTIAL、2 NO','3.13 ms 降到 2.01 ms','0.612–4.335 倍','kTile','106,496 B','1.588 倍','1.342','2.315','1.942','index to i64','memref.copy %arg0, %alloc','Pass Manager with 3 passes','2x64x32','arith.cmpi ult']:
    assert phrase in text, phrase
for bad in ['feature 层 15 族为 8、6、1','3.13 ms 到 1.02 ms','旧报告的标定摘要记为 1.36','98,304','131,072','约 30 倍','1.4–1.5×','约 27 µs 的装载与发射开销不含运行时调优','给定同一候选空间后','2^-17 的乘积类']:
    assert bad not in text,bad
# Every chapter has an incrementing ### sequence, including the reordered transform chapter.
for f in sorted(parts.glob('CH*.md')):
    nums=[int(x) for x in re.findall(r'^### (\d+)\. ',f.read_text(encoding='utf-8'),re.M)]
    assert nums==list(range(1,len(nums)+1)),(f.name,nums)
print('PASS: chapters=32, figures=12, markdown images=12, TOC links valid, assets exist, corrected facts present')
print('html_bytes',len(h.encode()),'md_bytes',len(text.encode()),'tables',p.tables,'h3',len(re.findall(r'^### ',text,re.M)))
print('sha256_html',hashlib.sha256(h.encode()).hexdigest())
print('sha256_md',hashlib.sha256(text.encode()).hexdigest())
