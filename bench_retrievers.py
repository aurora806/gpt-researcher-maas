"""Benchmark: multi-retriever latency, serial vs concurrent.

Reproduces the retrieval loop from `skills/researcher.py:832-864`
(`_search_relevant_source_urls`) and measures the serialization cost.

The first version of this benchmark was invalid: a single trial per mode let
per-call network variance (the same arxiv call measured 9.9s and 21.2s on two
runs) swamp the scheduling effect. This version separates the two concerns:

  Phase 1  latency profile   — N trials per (query, retriever), report median
  Phase 2  block comparison  — actual serial block vs concurrent block,
                               repeated, report median
  Phase 3  synthetic control — fixed-latency mock retrievers, which removes
                               network variance entirely and isolates the
                               scheduling effect. This is the mechanism proof.
  Phase 4  arxiv behaviour   — relevance on Chinese vs English queries

Predicted model:  serial ~= sum(median latencies), concurrent ~= max(median).

Usage:
    python bench_retrievers.py
"""
import asyncio
import statistics
import time

from dotenv import load_dotenv

load_dotenv()

from gpt_researcher.actions.retriever import get_retriever

# arxiv is excluded from the timing phases: its client enforces a 3s inter-request
# delay and arXiv rate-limits aggressively, producing 0.6s-21s swings. It is
# measured separately in phase 4.
FAST_RETRIEVERS = ["tavily", "openalex"]

MAX_RESULTS = 5  # cfg.max_search_results_per_query default

QUERIES = [
    "2026年新能源汽车购置税减半政策",
    "新能源汽车购置税减免政策要点",
    "2026年充电桩补贴政策",
    "electric vehicle purchase tax incentive policy China",
]

TRIALS_PROFILE = 3
TRIALS_BLOCK = 2
TRIALS_SYNTHETIC = 2
MOCK_DELAY = 2.0


async def timed_search(retriever_cls, query: str, kwargs: dict | None = None) -> dict:
    """Run one retriever and time it. Mirrors the shipped call shape."""
    t0 = time.perf_counter()
    try:
        retriever = retriever_cls(query, **(kwargs or {}))
        results = await asyncio.to_thread(retriever.search, max_results=MAX_RESULTS)
        return {"ok": True, "n": len(results), "sec": time.perf_counter() - t0,
                "results": results}
    except Exception as exc:
        return {"ok": False, "n": 0, "sec": time.perf_counter() - t0,
                "err": f"{type(exc).__name__}: {exc}", "results": []}


def median_seconds(trials: list[dict]) -> float:
    return statistics.median(t["sec"] for t in trials)


# ---------------------------------------------------------------------------
# Phase 3 support: a retriever with no network at all.
# ---------------------------------------------------------------------------
class MockRetriever:
    """Fixed-latency stand-in. Removes network variance from the comparison."""

    delay = MOCK_DELAY

    def __init__(self, query, query_domains=None):
        self.query = query

    def search(self, max_results=5):
        time.sleep(self.delay)
        return [{"href": f"https://example.invalid/{self.query}", "body": "x"}]


def make_mock(delay: float):
    return type(f"Mock{delay}", (MockRetriever,), {"delay": delay})


async def phase1_profile(retriever_classes, names):
    print("=" * 78)
    print("PHASE 1 — per-retriever latency profile (median of %d trials)" % TRIALS_PROFILE)
    print("=" * 78)
    profile = {}
    for q in QUERIES:
        print(f"\nQUERY: {q}")
        print(f"  {'retriever':<12} {'median':>8} {'min':>8} {'max':>8} {'hits':>6}")
        for name, cls in zip(names, retriever_classes):
            trials = [await timed_search(cls, q) for _ in range(TRIALS_PROFILE)]
            secs = [t["sec"] for t in trials]
            hits = trials[0]["n"]
            profile.setdefault(name, []).append(median_seconds(trials))
            print(f"  {name:<12} {statistics.median(secs):7.2f}s "
                  f"{min(secs):7.2f}s {max(secs):7.2f}s {hits:6d}")
    print()
    print("  per-retriever mean-of-medians:")
    for name in names:
        m = statistics.mean(profile[name])
        print(f"    {name:<12} {m:.2f}s")
    return profile


async def phase2_blocks(retriever_classes, names):
    print()
    print("=" * 78)
    print("PHASE 2 — actual block wall-clock (median of %d trials)" % TRIALS_BLOCK)
    print("=" * 78)

    async def serial_block(q):
        t0 = time.perf_counter()
        for cls in retriever_classes:
            await timed_search(cls, q)
        return time.perf_counter() - t0

    async def concurrent_block(q):
        t0 = time.perf_counter()
        await asyncio.gather(*[timed_search(cls, q) for cls in retriever_classes])
        return time.perf_counter() - t0

    print(f"\n  {'query':<44} {'serial':>9} {'conc':>9} {'speedup':>9}")
    totals = {"serial": [], "concurrent": []}
    for i, q in enumerate(QUERIES):
        # Alternate order so network drift does not systematically favour a mode.
        if i % 2 == 0:
            s = [await serial_block(q) for _ in range(TRIALS_BLOCK)]
            c = [await concurrent_block(q) for _ in range(TRIALS_BLOCK)]
        else:
            c = [await concurrent_block(q) for _ in range(TRIALS_BLOCK)]
            s = [await serial_block(q) for _ in range(TRIALS_BLOCK)]
        ms, mc = statistics.median(s), statistics.median(c)
        totals["serial"].append(ms)
        totals["concurrent"].append(mc)
        label = q if len(q) <= 42 else q[:41] + "…"
        print(f"  {label:<44} {ms:8.2f}s {mc:8.2f}s {ms / mc:8.2f}x")

    ms = statistics.mean(totals["serial"])
    mc = statistics.mean(totals["concurrent"])
    print(f"\n  {'MEAN':<44} {ms:8.2f}s {mc:8.2f}s {ms / mc:8.2f}x")
    return ms, mc


async def phase3_synthetic():
    print()
    print("=" * 78)
    print(f"PHASE 3 — synthetic control (3 mock retrievers, fixed {MOCK_DELAY}s each)")
    print("=" * 78)
    print("  No network. Any serial-vs-concurrent gap here is pure scheduling.\n")

    mocks = [make_mock(MOCK_DELAY) for _ in range(3)]

    serial_times, conc_times = [], []
    for _ in range(TRIALS_SYNTHETIC):
        t0 = time.perf_counter()
        for cls in mocks:
            await timed_search(cls, "q")
        serial_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        await asyncio.gather(*[timed_search(cls, "q") for cls in mocks])
        conc_times.append(time.perf_counter() - t0)

    ms, mc = statistics.median(serial_times), statistics.median(conc_times)
    print(f"  serial     : {ms:.2f}s   (expected {3 * MOCK_DELAY:.2f}s = 3 x {MOCK_DELAY})")
    print(f"  concurrent : {mc:.2f}s   (expected {MOCK_DELAY:.2f}s = max)")
    print(f"  speedup    : {ms / mc:.2f}x   (expected 3.00x)")
    return ms, mc


async def phase4_arxiv():
    print()
    print("=" * 78)
    print("PHASE 4 — arxiv relevance: Chinese vs English")
    print("=" * 78)
    cls = get_retriever("arxiv")
    for q in QUERIES:
        r = await timed_search(cls, q)
        status = f"{r['n']} hits in {r['sec']:.2f}s" if r["ok"] else f"ERR {r['err']}"
        print(f"  {q[:50]:<52} {status}")
        for hit in r["results"][:2]:
            print(f"      - {hit.get('title', '')[:70]}")


async def relevance_probe():
    print()
    print("=" * 78)
    print("openalex relevance probe (Chinese query)")
    print("=" * 78)
    cls = get_retriever("openalex")
    r = await timed_search(cls, QUERIES[0])
    print(f"  query: {QUERIES[0]}")
    for hit in r["results"][:5]:
        print(f"    - {hit.get('title', '')[:74]}")


async def main():
    retriever_classes = [get_retriever(n) for n in FAST_RETRIEVERS]
    for name, cls in zip(FAST_RETRIEVERS, retriever_classes):
        assert cls is not None, f"retriever {name!r} did not resolve"

    print(f"Timed retrievers: {FAST_RETRIEVERS}   max_results={MAX_RESULTS}\n")

    print("Warming up (DNS/TLS)...")
    for cls in retriever_classes:
        await timed_search(cls, "warmup")
    print("done.\n")

    await phase3_synthetic()
    profile = await phase1_profile(retriever_classes, FAST_RETRIEVERS)
    ms, mc = await phase2_blocks(retriever_classes, FAST_RETRIEVERS)

    # Model prediction from the measured per-retriever medians.
    pred_serial = sum(statistics.mean(profile[n]) for n in FAST_RETRIEVERS)
    pred_conc = max(statistics.mean(profile[n]) for n in FAST_RETRIEVERS)
    print()
    print("=" * 78)
    print("MODEL CHECK")
    print("=" * 78)
    print(f"  predicted from per-retriever medians: serial {pred_serial:.2f}s / "
          f"concurrent {pred_conc:.2f}s  ({pred_serial / pred_conc:.2f}x)")
    print(f"  measured block wall-clock          : serial {ms:.2f}s / "
          f"concurrent {mc:.2f}s  ({ms / mc:.2f}x)")

    await phase4_arxiv()
    await relevance_probe()


if __name__ == "__main__":
    asyncio.run(main())
