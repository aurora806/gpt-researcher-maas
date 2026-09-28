"""Probe: does the pipeline stop the model from fabricating citations?

Answer, from the code: it does not. The only constraint is one sentence in the
report prompt (gpt_researcher/prompts.py:310). There is no code that checks a
generated citation against what was actually retrieved:

  * add_references() (actions/markdown_processing.py:94) WOULD bind the
    reference list to the real `visited_urls` set, and it has a regression test
    (tests/test_add_references_order.py) - but the only caller is the
    detailed_report backend path (detailed_report.py:200), and it is applied to
    the *conclusion* only. The path measured here (research_report) never
    calls it.
  * SourceCurator ranks sources by relevance/credibility BEFORE the report
    exists, and is off by default (CURATE_SOURCES = False).
  * FactCheckerAgent lives in the multi_agents graph only, compares the draft
    against the context (not against URLs), and force-accepts once
    max_fact_check_revisions is exceeded.

So the question becomes empirical: what goes wrong in practice?

This probe reads finished reports and looks for citation defects that are
provable *without any external lookup*, using only internal contradiction:

  A. one domain carrying two different outlet labels   -> at most one is right
  B. one outlet label pointing at two different domains -> at most one is right
  C. a label naming a well-known outlet whose real domain is known to differ

(A) and (B) need no outside knowledge, so they cannot be argued away.

Usage:
    python probe_citations.py
"""
import glob
import os
import re
from collections import defaultdict
from urllib.parse import urlparse

LINK = re.compile(r"\[([^\]]*)\]\(\s*(https?://[^)\s]+)")
FRONTMATTER = re.compile(r"^---\n(.*?)\n---", re.S)
TAIL = ".,;:!?。，、）)]"

# Domains for outlets that are unambiguous. Used only for check C, which is
# reported separately from the self-contained checks A and B.
KNOWN = {
    "第一财经": "yicai.com",
    "证券时报": "stcn.com",
    "每日经济新闻": "nbd.com.cn",
    "中国消费者报": "ccnews.com.cn",
    "新华网": "news.cn",
    "人民网": "people.com.cn",
    "财联社": "cls.cn",
}


def norm_label(label: str) -> str:
    """'证券时报 (1)' -> '证券时报'; strip the '(Year)' suffix APA adds."""
    lab = re.sub(r"\s*\(\d+\)\s*$", "", label.strip())
    lab = re.sub(r"[,，]\s*\d{4}\s*$", "", lab)
    lab = re.sub(r"[等，,]\s*$", "", lab)
    return lab.strip()


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def collect(path):
    text = open(path, encoding="utf-8").read()
    fm = FRONTMATTER.search(text)
    sources_count = None
    if fm:
        m = re.search(r"sources_count:\s*(\d+)", fm.group(1))
        if m:
            sources_count = int(m.group(1))

    links = [(norm_label(lab), domain_of(u.rstrip(TAIL)))
             for lab, u in LINK.findall(text)]
    # Trailing reference lists often use [url](url); those are not attributions
    # and would otherwise show up as a second "label" on the same domain.
    links = [(lab, dom) for lab, dom in links if lab and not lab.startswith("http")]
    urls = {u.rstrip(TAIL) for _, u in LINK.findall(text)}
    return sources_count, links, urls


def main():
    files = [f for f in sorted(glob.glob("outputs/*.md"))
             if os.path.getsize(f) > 3000]
    if not files:
        print("No report files found in outputs/")
        return

    total_links = total_a = total_b = total_c = 0
    global_by_domain = defaultdict(set)   # domain -> labels, across ALL reports
    global_by_url = defaultdict(set)      # exact URL -> labels, across ALL reports

    for path in files:
        sources_count, links, urls = collect(path)
        if not links:
            continue
        for lab, dom in links:
            global_by_domain[dom].add(lab)
        for lab, u in LINK.findall(open(path, encoding="utf-8").read()):
            lab = norm_label(lab)
            if lab and not lab.startswith("http"):
                global_by_url[u.rstrip(TAIL)].add(lab)

        by_domain = defaultdict(set)
        by_label = defaultdict(set)
        for lab, dom in links:
            by_domain[dom].add(lab)
            by_label[lab].add(dom)

        # A: one domain, several outlet labels
        a = {d: labs for d, labs in by_domain.items() if len(labs) > 1}
        # B: one label, several domains
        b = {l: doms for l, doms in by_label.items() if len(doms) > 1}
        # C: known outlet, wrong domain
        c = [(l, d, KNOWN[l]) for l, d in {t for t in links}
             if l in KNOWN and KNOWN[l] not in d]

        total_links += len(links)
        total_a += sum(len(v) - 1 for v in a.values())
        total_b += sum(len(v) - 1 for v in b.values())
        total_c += len(c)

        print("=" * 78)
        print(os.path.basename(path))
        print("=" * 78)
        print(f"  sources_count (scraped URLs) : {sources_count}")
        print(f"  citation links               : {len(links)}")
        print(f"  distinct cited URLs          : {len(urls)}")
        if sources_count:
            print(f"  cited URLs vs scraped        : {len(urls)}/{sources_count}")
        print(f"  distinct cited domains       : {len(by_domain)}")
        print()

        if a:
            print("  [A] one domain, several outlet labels (at most one can be right)")
            for d, labs in sorted(a.items()):
                print(f"      {d}")
                for lab in sorted(labs):
                    print(f"          labeled as: {lab}")
        if b:
            print("  [B] one outlet label, several domains (at most one can be right)")
            for lab, doms in sorted(b.items()):
                print(f"      {lab} -> {', '.join(sorted(doms))}")
        if c:
            print("  [C] label names a known outlet, domain differs")
            for lab, got, want in sorted(c):
                print(f"      {lab:<16} cited as {got:<28} (expected {want})")
        if not (a or b or c):
            print("  No internal citation contradiction found in this report.")
        print()

    print("=" * 78)
    print("CROSS-REPORT: the same URL attributed to different outlets")
    print("=" * 78)
    print("  These runs scraped overlapping URLs. If attributions were derived from")
    print("  the source data they would agree; where they disagree, the attribution")
    print("  was produced by the model and cannot be traced to anything retrieved.")
    print()
    cross = {u: labs for u, labs in global_by_url.items() if len(labs) > 1}
    if cross:
        for u, labs in sorted(cross.items()):
            print(f"  {u}")
            for lab in sorted(labs):
                print(f"      attributed to: {lab}")
    else:
        print("  none found")
    print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  reports scanned        : {len(files)}")
    print(f"  citation links         : {total_links}")
    print(f"  [A] domain/label splits: {total_a}")
    print(f"  [B] label/domain splits: {total_b}")
    print(f"  [C] wrong-domain label : {total_c}")
    print()
    print("  Root cause is structural, not a matter of the model 'misbehaving':")
    print("  the context handed to the writer carries only Source/Title/Content")
    print("  (prompts.py:557) - no author and no publication date - while the")
    print("  report prompt requires APA-style (Author, Year) in-text citations")
    print("  (prompts.py:309, REPORT_FORMAT=APA). The attribution half of every")
    print("  citation therefore has to be invented to satisfy the format.")


if __name__ == "__main__":
    main()
