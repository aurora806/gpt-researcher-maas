"""独立测量脚本：跑一次政策调研任务，输出 6 项可写进简历的指标。

用法：
    python measure.py "你的政策主题"
"""
import asyncio
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from gpt_researcher import GPTResearcher  # noqa: E402

# 期望的"学术综述式"报告骨架（按改造目标定，当前默认 prompt 下仅作参考）
EXPECTED_SECTIONS = [
    "摘要", "研究背景", "政策演进", "政策要点",
    "现状", "问题", "比较", "结论", "建议", "参考文献",
]


async def main(query: str):
    researcher = GPTResearcher(query=query, report_type="research_report")

    # 1. 耗时：检索 + 写作 分段计时
    t0 = time.perf_counter()
    await researcher.conduct_research()
    t_research = time.perf_counter() - t0

    t0 = time.perf_counter()
    report = await researcher.write_report()
    t_write = time.perf_counter() - t0

    # 2. 源数量 / URL 去重数
    sources = researcher.get_research_sources()
    urls = researcher.get_source_urls()

    # 3. 源类别数（激活的检索器数）
    retrievers = list(getattr(researcher.cfg, "retrievers", []) or [])

    # 4. 成本
    cost = researcher.get_costs()
    step_costs = researcher.get_step_costs()

    # 5. 结构完整率（解析报告标题，与期望骨架比对）
    sections = researcher.extract_sections(report)
    headers = [s.get("section_title", "") for s in sections]
    found = sum(1 for e in EXPECTED_SECTIONS if any(e in h for h in headers))
    completeness = found / len(EXPECTED_SECTIONS) if EXPECTED_SECTIONS else 0.0

    # 6. 并发度（检索并发 = 检索器数；深研并发若配置存在则读取）
    deep_conc = getattr(researcher.cfg, "deep_research_concurrency", None)

    print()
    print("=" * 56)
    print(f"任务        : {query}")
    print(f"检索耗时    : {t_research:.1f}s")
    print(f"写作耗时    : {t_write:.1f}s")
    print(f"总耗时      : {t_research + t_write:.1f}s")
    print(f"信息源数量  : {len(sources)} 条 (去重URL {len(urls)} 个)")
    print(f"源类别数    : {len(retrievers)} 类 -> {retrievers}")
    print(f"API 成本    : ${cost:.4f}  (分步: {step_costs})")
    print(f"结构完整率  : {completeness*100:.0f}% ({found}/{len(EXPECTED_SECTIONS)})")
    print(f"检索并发度  : {len(retrievers)} (检索器并发)")
    if deep_conc:
        print(f"深研并发度  : {deep_conc} (Semaphore 上限)")
    if cost == 0.0:
        print("⚠ 成本为 0：DashScope 单价未映射到 cost_callback，")
        print("  成本指标需从阿里云控制台按 token 用量折算。")
    print("=" * 56)
    print()


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "2026年新能源汽车购车优惠政策"
    asyncio.run(main(q))
