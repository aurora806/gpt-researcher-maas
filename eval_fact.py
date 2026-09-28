"""Mini-FACT: do the citations in our own reports actually hold up?

Mirrors the FACT stage of DeepResearch Bench, in miniature and over our own
outputs instead of a public benchmark:

  1. extract every inline citation `[label](url)` from a finished report,
     together with the sentence it is attached to (the *claim*);
  2. re-fetch the cited page (with the project's own scraper);
  3. ask a judge whether the page actually supports the claim;
  4. report an effective-citation rate.

Why this is worth having: `sources_count` in the report frontmatter counts
*scraped* URLs, not *cited* ones, and nothing in the pipeline ever checks that
a cited URL supports the sentence it is attached to. This measures that.

Reported buckets per citation:

  supported     page was fetched and supports the claim
  unsupported   page was fetched and does NOT support the claim
  unfetchable   page could not be fetched (403 / timeout / blocked) - counted
                separately, never as "unsupported", because a site blocking us
                says nothing about whether the claim was right

Caching: fetched pages land in eval_fact_cache/ so re-runs are free. Judgements
land in eval_fact_results.json.

Usage:
    python eval_fact.py                 # all reports in outputs/
    python eval_fact.py --limit 5       # first 5 citations only (smoke test)
    python eval_fact.py --report outputs/2026新能源购置税减半.md
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

from gpt_researcher.actions.web_scraping import scrape_urls
from gpt_researcher.config.config import Config
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.utils.llm import create_chat_completion
from gpt_researcher.utils.workers import WorkerPool

CACHE_DIR = "eval_fact_cache"
RESULTS_PATH = "eval_fact_results.json"

# --- deterministic grounding check (no LLM) ---------------------------------
# Instead of asking a judge, check two things that must hold if the citation is
# honest: the specific figures in the claim have to appear on the cited page,
# and the outlet the citation names has to appear on it too. Both are cheap,
# reproducible, and free of judge bias. They are a *proxy*: a page can support a
# claim using different wording, and extraction can drop text, so a miss is a
# flag to look at, not a proof of fabrication.

FULLWIDTH = str.maketrans("０１２３４５６７８９．％", "0123456789.%")
YEAR = re.compile(r"^(?:19|20)\d{2}$")
NUM = re.compile(r"\d+(?:\.\d+)?")
# '1,000' tokenises as 1 and 000 under NUM, inventing a bogus missing figure
# ('000') and losing the real one. Drop the separator before matching.
THOUSANDS = re.compile(r"(?<=\d),(?=\d)")


def norm_digits(s: str) -> str:
    return THOUSANDS.sub("", s.translate(FULLWIDTH))

# Reports are written in simplified Chinese; some cited pages (e.g. Hong Kong
# government sites) are traditional. Without this, a correct attribution is
# flagged just because 环境及生态局 != 環境及生態局.
try:
    import zhconv as _zhconv

    def variants(s):
        return {s, _zhconv.convert(s, "zh-cn"), _zhconv.convert(s, "zh-tw")}
except Exception:  # zhconv not installed: degrade to exact match only
    def variants(s):
        return {s}


def _tokens(t: str):
    out = set()
    for m in NUM.finditer(t):
        s = m.group(0)
        if YEAR.match(s):
            continue
        if len(s.replace(".", "")) < 2:
            continue
        out.add(s.rstrip("0").rstrip(".") if "." in s else s)
    return out


def claim_numbers(claim: str):
    """Discriminating figures in the claim: years and single digits excluded."""
    return _tokens(norm_digits(claim))


# A report in one language may cite a page in the other: a claim saying
# '15 million' is backed by a page saying '1500万'. Containment alone calls
# that missing. Reduce both sides to a plain count of millions.
SCALED = re.compile(r"(\d+(?:\.\d+)?)\s*(million|billion|trillion|万|亿)", re.I)
_SCALE = {"million": 1.0, "billion": 1000.0, "trillion": 1e6, "万": 0.01, "亿": 100.0}


def million_values(t: str):
    """Every scaled quantity in t, in both of the forms a claim may use.

    '1500万' is '15 million' or '15000000'; either wording counts as a match,
    so emit both. Otherwise a correct citation is flagged purely because the
    two languages chose different units.
    """
    out = set()
    for m in SCALED.finditer(t):
        try:
            millions = float(m.group(1)) * _SCALE[m.group(2).lower()]
        except ValueError:
            continue
        out.add(f"{millions:g}")
        out.add(f"{millions * 1e6:g}")
    return out


def page_numbers(page: str):
    t = norm_digits(page)
    return _tokens(t) | million_values(t)


def in_order(label: str, page: str) -> bool:
    """Match an abbreviated outlet name against its full form.

    A report says 工信部 while the page says 工业和信息化部. Exact containment
    misses that; requiring the label's characters in order within a short
    window catches it. Only used for labels of >=3 chars, where a false
    positive is unlikely.
    """
    if len(label) < 3:
        return False
    return re.search(".{0,3}".join(re.escape(c) for c in label), page) is not None


def label_forms(label: str):
    """Labels look like '工业和信息化部' / '工信部' / '第一财经, 2025'."""
    lab = re.sub(r"[,，]\s*\d{4}\s*$", "", label.strip())
    lab = re.sub(r"\s*\(\d+\)\s*$", "", lab)
    lab = re.sub(r"[等，,]\s*$", "", lab)
    forms = [lab]
    # Try a shorter distinctive core too (e.g. 工业和信息化部 -> 工信部 won't
    # derive, but 中国太平洋保险集团 -> 太平洋保险 will).
    for n in (6, 5, 4, 3):
        if len(lab) > n:
            forms.append(lab[:n])
    return [f for f in forms if len(f) >= 3]


CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"^[\x20-\x7e]+$")


def latin_label_on_cjk_page(label: str, page: str) -> bool:
    """True when the outlet name is Latin script but the page is Chinese.

    A report written in English cites Chinese pages by their English names
    ('Xinhua Net' for news.cn). The page will never contain that string, so a
    miss here says nothing about whether the attribution is right or invented.
    Counting it as a failure would overstate the defect rate; counting it as a
    pass would hide one. It gets its own bucket.
    """
    return bool(LATIN.match(label)) and len(CJK.findall(page)) >= 200


def det_check(claim: str, label: str, page: str):
    nums = claim_numbers(claim)
    pnums = page_numbers(page)
    missing = sorted(n for n in nums if n not in pnums)

    lab_forms = label_forms(label)
    lab_hit = ""
    for f in lab_forms:
        hit = next((v for v in variants(f) if v in page), None)
        if hit:
            lab_hit = hit
            break
    if not lab_hit:
        for f in lab_forms:
            if in_order(f, page):
                lab_hit = f"{f} (abbreviated form on page)"
                break

    detail = []
    if missing:
        detail.append("figures not on page: " + ", ".join(missing))
    # A label too short to search for (<3 chars) is skipped rather than
    # counted against the citation.
    label_searched = bool(lab_forms)
    xlang = (label_searched and not lab_hit
             and latin_label_on_cjk_page(lab_forms[0], page))
    if label_searched and not lab_hit and not xlang:
        detail.append(f"outlet '{lab_forms[0]}' not on page")

    figures_ok = not missing
    if xlang and figures_ok:
        verdict = "unverifiable"
        detail.append(
            f"outlet '{lab_forms[0]}' is a Latin-script name for a Chinese "
            f"page: translation and invention look identical here"
        )
    elif not nums and not label_searched:
        verdict = "insufficient"
        detail.append("nothing checkable: no checkable figures, label too short")
    elif figures_ok and (lab_hit or not label_searched):
        verdict = "supported"
    elif missing and label_searched and not lab_hit:
        verdict = "unsupported"
    else:
        verdict = "insufficient"

    return {
        "verdict": verdict,
        "quote": f"outlet matched: {lab_hit}" if lab_hit else "",
        "reason": "; ".join(detail)[:300] or "all figures and the outlet appear on the page",
        "n_claim_numbers": len(nums),
        "n_missing": len(missing),
        "label_hit": lab_hit,
    }

FRONTMATTER = re.compile(r"^---\n(.*?)\n---", re.S)
CITE = re.compile(r"[（(]?\s*\[([^\]]{1,120})\]\(\s*(https?://[^)\s]+?)\s*\)\s*[）)]?")
SENT_SPLIT = re.compile(r"(?<=[。！？!?])")
# A URL may carry '?query=...' and the sentence splitter treats '?' as a full
# stop, cutting the URL in two. The citation then fails to match and vanishes
# from the tally altogether - an undercount, not a missing defect. Mask URLs
# before splitting and restore them per sentence.
URL_MASK = re.compile(r"https?://[^\s)）\]]+")
MASK = "\x00{}\x00"
CLAUSE_SPLIT = re.compile(r"(?<=[，,；;：:])")

# How much page text to give the judge. Pages are chunked and the chunks most
# similar to the claim are kept, so this is a budget, not a truncation point.
CONTENT_BUDGET = 9000

JUDGE_PROMPT = """You are verifying one citation in a research report.

CLAIM (the sentence the citation is attached to):
{claim}

SOURCE PAGE ({url}) - extracted text:
\"\"\"
{content}
\"\"\"

Question: does the source page actually support the claim above?

Answer with ONLY a JSON object, no other text:
{{"verdict": "supported" | "unsupported" | "insufficient",
  "quote": "<short verbatim quote from the page that decides it, or empty>",
  "reason": "<one sentence>"}}

Use "supported" only if the page contains evidence for what the claim asserts.
Use "unsupported" if the page contradicts the claim, or is about something else
entirely. Use "insufficient" if the page is on-topic but does not contain
enough to confirm or deny the specific assertion (this includes pages that
merely repeat the claim without evidence)."""


def cache_path(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


PUNCT_EDGE = r"[\s*_（）()\[\]【】，,；;：:。.、]+"
MIN_CLAIM = 12


def claims_in_sentence(sent: str):
    """Attribute each citation in a sentence to the prose that precedes it.

    A sentence often carries several citations::

        ...设计（[工信部, 2025](u1)；[腾讯新闻, 2025](u2)）。

    Splitting on separators and taking "the clause holding the citation" gives
    an empty claim for u1, because u1 sits right after a ；. Anything dropped
    that way would silently leave the denominator, which is exactly the kind of
    miss this tool exists to avoid. Instead, take the text since the previous
    citation; if that is only punctuation, the citation shares its neighbour's
    claim.
    """
    out = []
    last_end, last_claim = 0, ""
    for m in CITE.finditer(sent):
        gap = re.sub(PUNCT_EDGE + r"$", "", re.sub("^" + PUNCT_EDGE, "",
                                                   sent[last_end:m.start()]))
        claim = gap if len(gap) >= MIN_CLAIM else last_claim
        if len(claim) > 300:            # keep the judge's view tight
            parts = [p for p in CLAUSE_SPLIT.split(claim) if len(p) >= MIN_CLAIM]
            claim = parts[-1] if parts else claim[-300:]
        if len(claim) >= MIN_CLAIM:
            out.append((claim.strip(),
                        m.group(1).strip(),
                        m.group(2).rstrip(".,;:!?。，、）)]")))
            last_claim = claim.strip()
        last_end = m.end()
    return out


def mask_urls(text: str):
    """Replace every URL with a position token; return (masked, urls)."""
    urls = []

    def sub(m):
        urls.append(m.group(0))
        return MASK.format(len(urls) - 1)

    return URL_MASK.sub(sub, text), urls


def unmask_urls(text: str, urls):
    return re.sub(r"\x00(\d+)\x00", lambda m: urls[int(m.group(1))], text)


def extract_citations(report_text: str):
    """Return [(claim, url, label, frontmatter_sources_count)]."""
    fm = FRONTMATTER.search(report_text)
    sources_count = None
    if fm:
        m = re.search(r"sources_count:\s*(\d+)", fm.group(1))
        if m:
            sources_count = int(m.group(1))

    body = report_text[fm.end():] if fm else report_text
    # Drop the trailing reference list: entries there are [name](url) with no
    # claim attached, so they cannot be fact-checked this way.
    body = re.split(r"\n#{1,3}\s*(?:Reference|References|参考文献|参考资料)",
                    body)[0]

    out = []
    for line in body.split("\n"):
        if not CITE.search(line):
            continue
        masked, urls = mask_urls(line)
        for sent in SENT_SPLIT.split(masked):
            sent = unmask_urls(sent, urls)
            if not CITE.search(sent):
                continue
            for claim, label, url in claims_in_sentence(sent):
                out.append({"claim": claim, "url": url, "label": label,
                            "sources_count": sources_count})
    return out


def dedupe(cites):
    seen, out = set(), []
    for c in cites:
        key = (c["claim"], c["url"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def select_content(claim: str, page_text: str, embedder) -> str:
    """Keep the passages most relevant to the claim, within a char budget."""
    if len(page_text) <= CONTENT_BUDGET:
        return page_text
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=120).split_text(page_text)
    if not chunks:
        return page_text[:CONTENT_BUDGET]
    try:
        import numpy as np
        from langchain_community.utils.math import cosine_similarity
        cv = np.asarray(embedder.embed_documents(chunks), dtype=float)
        qv = np.asarray(embedder.embed_query(claim), dtype=float)
        sims = cosine_similarity([qv], cv)[0]
        order = sims.argsort()[::-1]
        kept, used = [], 0
        for i in order:
            if used + len(chunks[i]) > CONTENT_BUDGET and kept:
                break
            kept.append(chunks[i])
            used += len(chunks[i])
        return "\n...\n".join(kept)
    except Exception:
        return page_text[:CONTENT_BUDGET]


async def fetch_pages(urls, cfg):
    """Fetch each URL once; cache to disk. Returns {url: text or None}."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    pages, todo = {}, []
    for u in urls:
        p = cache_path(u)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    rec = json.load(fh)
                pages[u] = rec.get("text")
                continue
            except Exception:
                pass
        todo.append(u)

    if todo:
        pool = WorkerPool(cfg.max_scraper_workers, cfg.scraper_rate_limit_delay)
        print(f"  fetching {len(todo)} pages not in cache...")
        scraped, _ = await scrape_urls(todo, cfg, pool)
        got = {}
        for d in scraped:
            if d.get("raw_content"):
                got[d["url"]] = d["raw_content"]
        for u in todo:
            text = got.get(u)
            pages[u] = text
            with open(cache_path(u), "w", encoding="utf-8") as fh:
                json.dump({"url": u, "text": text}, fh, ensure_ascii=False)
    return pages


async def judge(claim, url, content, cfg):
    prompt = JUDGE_PROMPT.format(claim=claim, url=url, content=content)
    try:
        raw = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system",
                 "content": "You are a meticulous citation fact-checker."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
        )
    except Exception as exc:
        return {"verdict": "error", "quote": "", "reason": f"{type(exc).__name__}: {exc}"[:200]}

    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {"verdict": "error", "quote": "", "reason": f"unparseable: {raw!r}"[:200]}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {"verdict": "error", "quote": "", "reason": f"bad json: {m.group(0)!r}"[:200]}
    v = str(d.get("verdict", "")).lower()
    if v not in ("supported", "unsupported", "insufficient"):
        v = "error"
    return {"verdict": v, "quote": str(d.get("quote", ""))[:300],
            "reason": str(d.get("reason", ""))[:300]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true",
                    help="only fetch pages, skip the judge (dry run)")
    ap.add_argument("--judge", choices=["det", "llm"], default="det",
                    help="det = deterministic grounding check (free, no API); "
                         "llm = ask the configured model to judge")
    args = ap.parse_args()

    import glob
    reports = args.report or [
        f for f in sorted(glob.glob("outputs/*.md")) if os.path.getsize(f) > 3000
    ]

    cites, per_report = [], {}
    for r in reports:
        text = open(r, encoding="utf-8").read()
        got = dedupe(extract_citations(text))
        for c in got:
            c["report"] = os.path.basename(r)
        per_report[os.path.basename(r)] = {
            "citations": len(got),
            "distinct_urls": len({c["url"] for c in got}),
            "sources_count": got[0]["sources_count"] if got else None,
        }
        cites.extend(got)

    print("=" * 78)
    print("EXTRACTED CITATIONS")
    print("=" * 78)
    for name, s in per_report.items():
        print(f"  {name:<44} citations={s['citations']:>3} "
              f"urls={s['distinct_urls']:>3} scraped={s['sources_count']}")
    print(f"  TOTAL checkable citations: {len(cites)}")
    print(f"  TOTAL distinct URLs      : {len({c['url'] for c in cites})}")

    if args.limit:
        cites = cites[:args.limit]
        print(f"  (limited to first {len(cites)})")

    cfg = Config()
    urls = sorted({c["url"] for c in cites})
    print()
    print("=" * 78)
    print("FETCHING CITED PAGES")
    print("=" * 78)
    pages = await fetch_pages(urls, cfg)
    ok = sum(1 for u in urls if pages.get(u))
    print(f"  fetched {ok}/{len(urls)} distinct cited URLs")

    if args.no_judge:
        print("\n--no-judge: stopping before the judge.")
        return

    embedder = None
    if args.judge == "llm":
        embedder = Memory(cfg.embedding_provider, cfg.embedding_model).get_embeddings()

    print()
    print("=" * 78)
    print(f"JUDGING  (mode: {args.judge})")
    print("=" * 78)
    results = []
    for i, c in enumerate(cites, 1):
        text = pages.get(c["url"])
        if not text:
            r = {"verdict": "unfetchable", "quote": "", "reason": "page not fetched"}
        elif args.judge == "det":
            r = det_check(c["claim"], c["label"], text)
        else:
            content = select_content(c["claim"], text, embedder)
            r = await judge(c["claim"], c["url"], content, cfg)
        row = {**c, **r}
        results.append(row)
        mark = {"supported": "OK  ", "unsupported": "FAIL", "insufficient": "WEAK",
                "unfetchable": "N/A ", "error": "ERR "}.get(r["verdict"], "?   ")
        print(f"  [{i:>2}/{len(cites)}] {mark} {c['label'][:22]:<22} {c['url'][:52]}")
        if r["verdict"] in ("unsupported", "insufficient") and r.get("reason"):
            print(f"          claim : {c['claim'][:96]}")
            print(f"          reason: {r['reason'][:96]}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    out_path = RESULTS_PATH

    # --- summary ---
    print()
    print("=" * 78)
    print("EFFECTIVE CITATION RATE")
    print("=" * 78)
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(results)
    for k in ("supported", "unsupported", "insufficient", "unverifiable",
              "unfetchable", "error"):
        if counts.get(k):
            print(f"  {k:<14} {counts[k]:>3}  ({100 * counts[k] / total:5.1f}%)")
    print(f"  {'TOTAL':<14} {total:>3}")

    verifiable = total - counts.get("unfetchable", 0) - counts.get("error", 0)
    if verifiable:
        print()
        print(f"  effective citation rate (supported / fetched) = "
              f"{counts.get('supported', 0)}/{verifiable} = "
              f"{100 * counts.get('supported', 0) / verifiable:.1f}%")

    if args.judge == "det":
        n_num = sum(r.get("n_claim_numbers", 0) for r in results)
        n_miss = sum(r.get("n_missing", 0) for r in results)
        n_lab = sum(1 for r in results if r.get("label_hit"))
        print()
        print("  deterministic detail:")
        print(f"    figures checked on cited pages : {n_num} "
              f"({n_miss} absent from the page)")
        print(f"    citations whose outlet name appears on the page: "
              f"{n_lab}/{total - counts.get('unfetchable', 0)}")
    print(f"\n  details -> {out_path}")

    # per-report breakdown
    print()
    print("  per report:")
    by_rep = {}
    for r in results:
        by_rep.setdefault(r["report"], []).append(r)
    for name, rows in by_rep.items():
        sup = sum(1 for r in rows if r["verdict"] == "supported")
        print(f"    {name:<44} {sup}/{len(rows)} supported")


if __name__ == "__main__":
    asyncio.run(main())
