#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试题音频预生成脚本
====================
从 面试题_可播放.html 提取每道题（.qbody）的纯文本，用微软 Edge 神经网络语音
（edge-tts，免费、无需 API Key）提前合成为 MP3。

用法：
  python gen_audio.py test                       # 连通性自检（生成 1 条样例）
  python gen_audio.py samples                    # 生成 晓晓/晓伊/瑶瑶 样音到 samples/
  python gen_audio.py all --voice zh-CN-XiaoxiaoNeural   # 批量生成 297 段到 面试题_音频/

特性：
  - 已存在的文件自动跳过（可断点续生成 / 重跑不重复）
  - 限并发 + 失败重试，稳定批量
  - 输出 manifest.json（id -> {voice,text_len,file}）
"""
import argparse
import asyncio
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "面试题_可播放.html")
AUDIO_DIR = os.path.join(HERE, "面试题_音频")
SAMPLES_DIR = os.path.join(HERE, "samples")

SAMPLE_IDS = ["m1q1", "m6q1"]  # 短题 + 长题，用于音色对比
VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # 晓晓：旗舰中文女声，温暖自然
    "xiaoyi": "zh-CN-XiaoyiNeural",      # 晓伊：清亮知性
    "yaoyao": "zh-CN-YaoyaoNeural",       # 瑶瑶：年轻活泼
}
DEFAULT_VOICE = VOICES["xiaoxiao"]
CONCURRENCY = 8
MAX_RETRY = 3


# ---------------------------------------------------------------------------
# 1) 解析：提取每道题 id -> 纯文本
# ---------------------------------------------------------------------------
class QuestionExtractor(HTMLParser):
    """遍历 HTML，提取每个 <div class="q" id="mXqY"> 内 .qbody 的纯文本。"""

    BLOCK_TAGS = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = {}          # id -> text
        self.cur_id = None
        self.capturing = False     # 是否处于 .qbody 内部
        self.qbody_depth = 0       # .qbody 内的 div 嵌套深度
        self.buf = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = (attrs_d.get("class") or "").split()
        if tag == "div":
            if "qbody" in cls:
                # 进入 qbody：开始捕获
                self.capturing = True
                self.qbody_depth = 1
                self.buf = []
            elif self.capturing:
                self.qbody_depth += 1
            # 记录当前题 id（.q 容器）
            if "q" in cls and "qbody" not in cls and attrs_d.get("id"):
                self.cur_id = attrs_d["id"]
        if self.capturing and tag in self.BLOCK_TAGS:
            self.buf.append("\n")

    def handle_data(self, data):
        if self.capturing:
            self.buf.append(data)

    def handle_endtag(self, tag):
        if self.capturing and tag == "div":
            self.qbody_depth -= 1
            if self.qbody_depth <= 0:
                text = _clean_text("".join(self.buf))
                if self.cur_id:
                    self.results[self.cur_id] = text
                self.capturing = False
                self.buf = []
                self.cur_id = None


def _clean_text(raw: str) -> str:
    """解码实体、折叠空白，保留中文标点以便 TTS 自然断句。"""
    txt = html.unescape(raw)
    # 把换行/多余空白折叠为单个空格，但保留已有标点
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def extract_questions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parser = QuestionExtractor()
    parser.feed(content)
    return parser.results


# ---------------------------------------------------------------------------
# 2) 合成：edge-tts
# ---------------------------------------------------------------------------
async def _gen_one(text: str, path: str, voice: str):
    """生成单条音频到 path；已存在则跳过。返回 True 表示新生成。"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return False
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            comm = edge_tts.Communicate(text, voice)
            await comm.save(path)
            return True
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(1.5 * attempt)
    raise RuntimeError(f"生成失败 {path}: {last_err}")


async def _gen_batch(items, out_dir, voice, concurrency=CONCURRENCY, progress=None):
    """items: list[(id, text)]；并发生成，跳过已存在。"""
    os.makedirs(out_dir, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(items)

    async def worker(item):
        nonlocal done
        qid, text = item
        path = os.path.join(out_dir, qid + ".mp3")
        async with sem:
            await _gen_one(text, path, voice)
        done += 1
        if progress and done % 20 == 0 or done == total:
            progress(done, total)

    await asyncio.gather(*(worker(it) for it in items))


# ---------------------------------------------------------------------------
# 3) 子命令
# ---------------------------------------------------------------------------
def cmd_test():
    qs = extract_questions(HTML_PATH)
    if not qs:
        print("未提取到任何题目，请检查 HTML 结构。", file=sys.stderr)
        return 1
    qid = next(iter(qs))
    print(f"提取到 {len(qs)} 道题。连通性自检：合成 {qid} ...")
    out = os.path.join(SAMPLES_DIR, "test.mp3")
    asyncio.run(_gen_one(qs[qid], out, DEFAULT_VOICE))
    size = os.path.getsize(out)
    print(f"OK -> {out} ({size} bytes)。Edge TTS 端点可访问。")
    return 0


def cmd_samples():
    qs = extract_questions(HTML_PATH)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print(f"生成样音（短题 {SAMPLE_IDS[0]} + 长题 {SAMPLE_IDS[1]}）x {len(VOICES)} 音色 ...")
    async def run():
        tasks = []
        for vkey, vname in VOICES.items():
            for qid in SAMPLE_IDS:
                out = os.path.join(SAMPLES_DIR, f"{vkey}_{qid}.mp3")
                tasks.append(_gen_one(qs[qid], out, vname))
        await asyncio.gather(*tasks)
    asyncio.run(run())
    print("样音已生成到 samples/：")
    for f in sorted(os.listdir(SAMPLES_DIR)):
        if f.endswith(".mp3"):
            print("  ", f)
    return 0


def cmd_all(voice: str):
    qs = extract_questions(HTML_PATH)
    print(f"批量生成 {len(qs)} 段音频 -> {AUDIO_DIR}（音色 {voice}）...")
    items = list(qs.items())

    def progress(done, total):
        print(f"  进度 {done}/{total}", end="\r", flush=True)

    asyncio.run(_gen_batch(items, AUDIO_DIR, voice, progress=progress))
    print()

    # 写清单
    manifest = {qid: {"voice": voice, "text_len": len(text)} for qid, text in qs.items()}
    with open(os.path.join(AUDIO_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    sizes = [os.path.getsize(os.path.join(AUDIO_DIR, qid + ".mp3")) for qid in qs]
    total_mb = sum(sizes) / 1024 / 1024
    print(f"完成：{len(sizes)} 个文件，总体积 {total_mb:.2f} MB，平均 {sum(sizes)//len(sizes)//1024} KB/题。")
    print(f"清单 -> {os.path.join(AUDIO_DIR, 'manifest.json')}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="面试题音频预生成（edge-tts）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test", help="连通性自检，生成 1 条样例")
    sub.add_parser("samples", help="生成 晓晓/晓伊/瑶瑶 样音到 samples/")
    p_all = sub.add_parser("all", help="批量生成全部题目")
    p_all.add_argument("--voice", default=DEFAULT_VOICE,
                       help=f"音色 voice 名（默认 {DEFAULT_VOICE}）")
    args = ap.parse_args()

    if args.cmd == "test":
        return cmd_test()
    if args.cmd == "samples":
        return cmd_samples()
    if args.cmd == "all":
        return cmd_all(args.voice)
    return 1


if __name__ == "__main__":
    sys.exit(main())
