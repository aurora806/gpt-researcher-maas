"""Probe: what does each configured retriever actually contribute?

Reproduces the aggregation in `skills/researcher.py:824-870`
(`_search_relevant_source_urls`): iterate retrievers **in config order**, call
`search(max_results)`, collect URLs, then dedupe through `_get_new_urls`.

Because dedup runs after the loop, the retriever listed first structurally wins
ties — `RETRIEVER=tavily,arxiv,openalex` gives tavily first claim on every URL.
This probe measures that effect rather than assuming it.

Usage:
    python probe_multisource.py
"""
import asyncio
from collections import Counter
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from gpt_researcher.actions.retriever import get_retriever

RETRIEVER_NAMES = ["tavily", "arxiv", "openalex"]
MAX_RESULTS = 5

QUERIES = [
    "2026年新能源汽车购置税减半政策",
    "新能源汽车购置税减免政策要点",
    "2026年充电桩补贴政策",
]

ACADEMIC_HINTS = (
    "arxiv.org", "doi.org", "springer", "sciencedirect", "ieee",
    "nature.com", "mdpi.com", "researchgate", "semanticscholar",
    "openalex.org", "ncbi.nlm.nih.gov", "tandfonline", "wiley",
)


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return "?"


def is_academic(url: str) -> bool:
    return any(h in url.lower() for h in ACADEMIC_HINTS)


async def collect(retriever_cls, query):
    """Mirrors the shipped call shape."""
    try:
        r = retriever_cls(query, query_domains=None)
        return await asyncio.to_thread(r.search, max_results=MAX_RESULTS)
    except Exception as exc:
        print(f"    [{retriever_cls.__name__}] ERROR {type(exc).__name__}: {exc}")
        return []


async def main():
    classes = [get_retriever(n) for n in RETRIEVER_NAMES]
    print(f"Config order: RETRIEVER={','.join(RETRIEVER_NAMES)}  max_results={MAX_RESULTS}\n")

    grand_raw = Counter()
    grand_kept = Counter()
    grand_academic_kept = Counter()

    for q in QUERIES:
        print("=" * 78)
        print(f"QUERY: {q}")
        print("=" * 78)

        # --- pass 1: raw yield per retriever, before any dedup ---
        raw = {}
        for name, cls in zip(RETRIEVER_NAMES, classes):
            res = await collect(cls, q)
            urls = [r.get("href") or r.get("url") for r in res if isinstance(r, dict)]
            urls = [u for u in urls if u]
            raw[name] = urls
            grand_raw[name] += len(urls)
            print(f"  raw  {name:<10} {len(urls):>2} urls")

        # --- pass 2: shipped dedup order (visited_urls shared, config order) ---
        visited = set()
        kept = {}
        for name in RETRIEVER_NAMES:
            new = []
            for u in raw[name]:
                if u not in visited:
                    visited.add(u)
                    new.append(u)
            kept[name] = new
            grand_kept[name] += len(new)
            grand_academic_kept[name] += sum(1 for u in new if is_academic(u))

        print(f"  {'':<6} {'raw':>4} {'kept':>6} {'overlap':>8}   domains kept")
        for name in RETRIEVER_NAMES:
            overlap = len(raw[name]) - len(kept[name])
            doms = Counter(domain(u) for u in kept[name]).most_common(3)
            dom_str = ", ".join(f"{d}({c})" for d, c in doms) or "-"
            print(f"  {name:<10} {len(raw[name]):>4} {len(kept[name]):>6} {overlap:>8}   {dom_str}")

        total_kept = sum(len(v) for v in kept.values())
        academic = sum(1 for v in kept.values() for u in v if is_academic(u))
        print(f"  TOTAL kept: {total_kept}  (of which academic-looking: {academic})")
        print()

    print("=" * 78)
    print("AGGREGATE over %d queries" % len(QUERIES))
    print("=" * 78)
    print(f"  {'retriever':<12} {'raw':>6} {'kept':>8} {'kept %':>9} {'academic':>10}")
    total_raw = sum(grand_raw.values()) or 1
    total_kept = sum(grand_kept.values()) or 1
    for name in RETRIEVER_NAMES:
        r, k = grand_raw[name], grand_kept[name]
        print(f"  {name:<12} {r:>6} {k:>8} {100 * k / total_raw:8.1f}% "
              f"{grand_academic_kept[name]:>10}")
    print(f"  {'TOTAL':<12} {total_raw:>6} {total_kept:>8}")
    print(f"\n  of the {total_kept} URLs that survive dedup, "
          f"{sum(grand_academic_kept.values())} look academic")


if __name__ == "__main__":
    asyncio.run(main())
