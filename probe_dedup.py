"""Probe: how the context pipeline actually filters, and whether 0.42 is calibrated.

Reproduces `ContextCompressor` (gpt_researcher/context/compression.py:127-188):

    RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
      -> EmbeddingsFilter(k=20 DEFAULT, similarity_threshold=cfg.similarity_threshold)
      -> pretty_print_docs(docs, max_results=10)

`EmbeddingsFilter.compress_documents` (langchain_classic) does:

    included = argsort(similarity)[::-1][:k]          # top-k, k defaults to 20
    included = included[similarity[included] > thr]   # STRICTLY greater

and `EmbeddingsFilter.similarity_fn` defaults to
`langchain_community.utils.math.cosine_similarity`.

The point of this probe: SIMILARITY_THRESHOLD defaults to 0.42 in
config/variables/default.py:6, a value inherited from OpenAI embedding models.
Cosine-similarity distributions are model-specific, so swapping in
qwen3.7-text-embedding without re-checking the threshold may silently drop
everything (or nothing). This measures the real distribution.

Usage:
    python probe_dedup.py
"""
import asyncio
import statistics

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

URLS = [
    "https://news.qq.com/rain/a/20251226A04PO100",
    "https://www.mof.gov.cn/jrttts/202306/t20230629_3893518.htm",
    "https://www.yicai.com/news/102819401.html",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
K_CAP = 20  # EmbeddingsFilter.k default
THRESHOLD = 0.42  # SIMILARITY_THRESHOLD default


def describe(name, arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        print(f"  {name:<28} (empty)")
        return
    print(f"  {name:<28} n={arr.size:<4} "
          f"min={arr.min():.4f}  p25={np.percentile(arr, 25):.4f}  "
          f"median={np.median(arr):.4f}  p75={np.percentile(arr, 75):.4f}  "
          f"max={arr.max():.4f}")


async def main():
    cfg = Config()
    print(f"embedding      : {cfg.embedding_provider}:{cfg.embedding_model}")
    print(f"similarity_thr : {cfg.similarity_threshold}")
    print(f"compression_thr: {8000}  (COMPRESSION_THRESHOLD default)")
    print(f"query          : {QUERY}\n")

    # --- 1. real scraped content through the project's own scraper ---
    pool = WorkerPool(cfg.max_scraper_workers, cfg.scraper_rate_limit_delay)
    scraped, _ = await scrape_urls(URLS, cfg, pool)
    scraped = [d for d in scraped if d.get("raw_content")]
    print(f"scraped pages  : {len(scraped)} of {len(URLS)} urls")
    for d in scraped:
        print(f"    {len(d['raw_content']):>7} chars  {d['url'][:64]}")
    total_chars = sum(len(d["raw_content"]) for d in scraped)
    print(f"total raw chars: {total_chars}")
    print(f"compression fast-path (total < 8000 and docs <= 10): "
          f"{'YES - embedding filter SKIPPED' if total_chars < 8000 else 'no - full pipeline runs'}\n")

    # --- 2. chunk exactly like ContextCompressor.__get_contextual_retriever ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for d in scraped:
        for chunk in splitter.split_documents(
            [Document(page_content=d["raw_content"], metadata={"source": d["url"]})]
        ):
            docs.append(chunk)
    texts = [d.page_content for d in docs]
    print(f"chunks         : {len(docs)} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # --- 3. embed with the configured model (BatchedOpenAIEmbeddings) ---
    emb = Memory(cfg.embedding_provider, cfg.embedding_model).get_embeddings()
    doc_vecs = np.asarray(emb.embed_documents(texts), dtype=float)
    q_vec = np.asarray(emb.embed_query(QUERY), dtype=float)
    print(f"vector dim     : {doc_vecs.shape[1]}")
    norms = np.linalg.norm(doc_vecs, axis=1)
    print(f"doc vec norms  : min={norms.min():.4f} median={np.median(norms):.4f} "
          f"max={norms.max():.4f}  -> "
          f"{'NORMALIZED (cosine == dot product)' if np.allclose(norms, norms[0], atol=1e-3) else 'NOT normalized'}")
    print()

    # --- 4. query relevance, exactly as EmbeddingsFilter computes it ---
    sims = cosine_similarity([q_vec], doc_vecs)[0]
    print("=" * 78)
    print("QUERY-TO-CHUNK COSINE SIMILARITY (what EmbeddingsFilter thresholds)")
    print("=" * 78)
    describe("similarity", sims)

    order = np.argsort(sims)[::-1]
    top_k = order[:K_CAP]
    passed_k = top_k[sims[top_k] > THRESHOLD]

    print(f"\n  step 1  top-k (k={K_CAP} default)     : {len(top_k)} chunks")
    print(f"  step 2  then > {THRESHOLD} (strict)      : {len(passed_k)} chunks survive")
    print(f"  step 3  pretty_print_docs(max=10)    : {min(len(passed_k), 10)} chunks in context")

    n_pass_all = int((sims > THRESHOLD).sum())
    print(f"\n  for reference, ignoring the k-cap: {n_pass_all}/{len(sims)} "
          f"chunks are > {THRESHOLD} ({100 * n_pass_all / len(sims):.1f}%)")
    if len(passed_k) == 0:
        print("\n  *** ZERO chunks pass. The context for this sub-query would be EMPTY. ***")
    elif len(passed_k) < 3:
        print(f"\n  *** Only {len(passed_k)} chunk(s) pass - very thin context. ***")

    # --- 5. what a redundancy filter WOULD catch (it is not used) ---
    print()
    print("=" * 78)
    print("PAIRWISE CHUNK SIMILARITY (EmbeddingsRedundantFilter territory - NOT USED)")
    print("=" * 78)
    if len(doc_vecs) > 1:
        pairwise = cosine_similarity(doc_vecs, doc_vecs)
        iu = np.triu_indices(len(doc_vecs), k=1)
        off = pairwise[iu]
        describe("pairwise sim", off)
        for t in (0.90, 0.95, 0.98):
            print(f"    pairs > {t:.2f}: {int((off > t).sum())}/{off.size}")

    # --- 6. what the threshold would have to be to keep a target count ---
    print()
    print("=" * 78)
    print("THRESHOLD SENSITIVITY (chunks surviving, ignoring k-cap)")
    print("=" * 78)
    for t in (0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60):
        n = int((sims > t).sum())
        bar = "#" * n
        print(f"  > {t:.2f}  {n:>3} chunks  {bar}")


if __name__ == "__main__":
    asyncio.run(main())
