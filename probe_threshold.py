"""Probe: does SIMILARITY_THRESHOLD=0.42 actually discriminate on-topic from
off-topic content, under qwen3.7-text-embedding?

probe_dedup.py showed that on-topic chunks score 0.55-0.89, so all of them pass
0.42. That alone does not prove the threshold is inert - a threshold only needs
to reject OFF-topic content, not on-topic content. This probe adds off-topic
controls to find out whether it does.

Corpus:
  ON-TOPIC  - three real pages cited by an actual run (NEV purchase-tax policy)
  OFF-TOPIC - real pages on unrelated subjects, best effort; plus clearly
              labelled synthetic Chinese prose as a guaranteed control, so the
              comparison still completes if the unrelated pages fail to scrape.

Usage:
    python probe_threshold.py
"""
import asyncio

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from langchain_community.utils.math import cosine_similarity
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gpt_researcher.actions.web_scraping import scrape_urls
from gpt_researcher.config.config import Config
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.utils.workers import WorkerPool

QUERY = "2026年新能源汽车购置税减半政策要点"
THRESHOLD = 0.42
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

ON_TOPIC_URLS = [
    "https://news.qq.com/rain/a/20251226A04PO100",
    "https://www.mof.gov.cn/jrttts/202306/t20230629_3893518.htm",
    "https://www.yicai.com/news/102819401.html",
]

OFF_TOPIC_URLS = [
    "https://baike.baidu.com/item/%E5%B7%9D%E8%8F%9C",
    "https://zh.wikipedia.org/wiki/%E7%B1%B3%E9%A5%AD",
]

# Guaranteed control: clearly unrelated Chinese prose, same register and length
# as the on-topic pages. Labelled as synthetic wherever it appears.
SYNTHETIC_OFF_TOPIC = [
    "川菜是中国八大菜系之一，起源于四川地区，以麻、辣、鲜、香为特色。川菜善于运用辣椒、花椒、豆瓣酱等调料，"
    "烹调方法多样，包括炒、爆、煸、烧、蒸、炸等三十余种。经典菜品有麻婆豆腐、回锅肉、宫保鸡丁、水煮鱼等，"
    "在国内外的餐饮市场都有广泛的影响力和受众基础。近年来，川菜产业规模持续扩大，连锁经营和标准化生产成为趋势。",
    "水稻是世界主要粮食作物之一，亚洲地区贡献了全球九成以上的产量。水稻种植需要充足的水源和适宜的温度，"
    "主要分为籼稻和粳稻两个亚种。现代农业技术通过良种选育、机械化插秧、病虫害综合防治等手段显著提高单产。"
    "米饭的烹饪方式因地域而异，蒸煮是最普遍的做法，不同品种的米在口感和黏性上差别明显。",
    "篮球运动起源于美国，由詹姆斯·奈史密斯于十九世纪末发明。比赛双方各五名球员，目标是将球投入对方篮筐。"
    "现代职业篮球联赛如NBA、CBA具有高度商业化和全球化的特征，球员转会、工资帽、选秀制度构成完整的运营体系。"
    "近年来数据分析在战术制定和球员评估中的应用越来越广泛，改变了传统的比赛策略。",
]


async def main():
    cfg = Config()
    pool = WorkerPool(cfg.max_scraper_workers, cfg.scraper_rate_limit_delay)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    emb = Memory(cfg.embedding_provider, cfg.embedding_model).get_embeddings()

    print(f"embedding : {cfg.embedding_provider}:{cfg.embedding_model}")
    print(f"threshold : {THRESHOLD}\n")

    def chunkify(records):
        out = []
        for text, src in records:
            out.extend(splitter.split_documents(
                [Document(page_content=text, metadata={"source": src})]
            ))
        return out

    # --- on-topic: real scraped pages ---
    scraped, _ = await scrape_urls(ON_TOPIC_URLS, cfg, pool)
    on_records = [(d["raw_content"], d["url"]) for d in scraped if d.get("raw_content")]
    on_chunks = chunkify(on_records)

    # --- off-topic: real pages, best effort ---
    off_scraped, _ = await scrape_urls(OFF_TOPIC_URLS, cfg, pool)
    off_real = [(d["raw_content"], d["url"]) for d in off_scraped if d.get("raw_content")]
    off_real_chunks = chunkify(off_real)

    off_syn_chunks = chunkify([(t, "synthetic://off-topic") for t in SYNTHETIC_OFF_TOPIC])

    print(f"ON-TOPIC  : {len(on_records)} real pages -> {len(on_chunks)} chunks")
    print(f"OFF-TOPIC : {len(off_real)} real pages -> {len(off_real_chunks)} chunks")
    print(f"            {len(SYNTHETIC_OFF_TOPIC)} synthetic docs -> {len(off_syn_chunks)} chunks")
    if not off_real_chunks:
        print("            (unrelated real pages did not scrape; synthetic control only)")
    print()

    q_vec = np.asarray(emb.embed_query(QUERY), dtype=float)

    def sims_for(chunks):
        if not chunks:
            return np.array([])
        vecs = np.asarray(emb.embed_documents([c.page_content for c in chunks]), dtype=float)
        return cosine_similarity([q_vec], vecs)[0]

    on_sims = sims_for(on_chunks)
    off_real_sims = sims_for(off_real_chunks)
    off_syn_sims = sims_for(off_syn_chunks)

    print("=" * 78)
    print("COSINE SIMILARITY TO THE POLICY QUERY")
    print("=" * 78)
    for label, arr in [("ON-TOPIC (real)", on_sims),
                       ("OFF-TOPIC (real)", off_real_sims),
                       ("OFF-TOPIC (synthetic)", off_syn_sims)]:
        if arr.size == 0:
            print(f"  {label:<24} (none)")
            continue
        print(f"  {label:<24} n={arr.size:<3} min={arr.min():.4f} "
              f"median={np.median(arr):.4f} max={arr.max():.4f}")

    print()
    print("=" * 78)
    print(f"DISCRIMINATION AT THRESHOLD = {THRESHOLD}")
    print("=" * 78)
    for label, arr in [("ON-TOPIC (real)", on_sims),
                       ("OFF-TOPIC (real)", off_real_sims),
                       ("OFF-TOPIC (synthetic)", off_syn_sims)]:
        if arr.size == 0:
            continue
        kept = int((arr > THRESHOLD).sum())
        mark = "  <-- should be ~0%" if "OFF" in label else "  <-- should be ~100%"
        print(f"  {label:<24} kept {kept:>2}/{arr.size:<3} "
              f"({100 * kept / arr.size:5.1f}%){mark}")

    # --- overlap: this is the number that matters ---
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    off_all = np.concatenate([a for a in (off_real_sims, off_syn_sims) if a.size])
    if off_all.size and on_sims.size:
        on_min, off_min, off_max = on_sims.min(), off_all.min(), off_all.max()
        separable = off_max < on_min
        print(f"  lowest on-topic score : {on_min:.4f}")
        print(f"  highest off-topic score: {off_max:.4f}")
        print(f"  off-topic that pass {THRESHOLD}: "
              f"{int((off_all > THRESHOLD).sum())}/{off_all.size}")
        print()
        if separable:
            print("  The two populations are separable -> a threshold exists that works,")
            print(f"  but {THRESHOLD} must be raised: need > {off_max:.3f}.")
        else:
            print("  The populations OVERLAP -> no single threshold separates them cleanly.")
        if (off_all > THRESHOLD).all():
            print(f"  {THRESHOLD} is INERT: it rejects nothing, including off-topic content.")
        elif (on_sims > THRESHOLD).all() and (off_all <= THRESHOLD).all():
            print(f"  {THRESHOLD} works correctly on this sample.")

    # --- what a workable threshold would be ---
    print()
    print("=" * 78)
    print("THRESHOLD SENSITIVITY (kept count per population)")
    print("=" * 78)
    print(f"  {'thr':>6} {'on-topic':>12} {'off-topic':>12}")
    for t in (0.42, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        ko = int((on_sims > t).sum())
        kf = int((off_all > t).sum())
        print(f"  {t:>6.2f} {ko:>7}/{len(on_sims):<4} {kf:>7}/{len(off_all):<4}")


if __name__ == "__main__":
    asyncio.run(main())
