"""Probe: is the embedding adaptation a CHINESE problem or an API problem?

The resume claim is "完成中文 embedding 适配". That claim is only true if the
failure is specific to Chinese text. If the same failure happens with English
text, then what was actually fixed is DashScope API compatibility, and the
"Chinese" framing is wrong.

The two changes in gpt_researcher/memory/embeddings.py are:
  1. check_embedding_ctx_length=False   (line 139/152)
  2. BatchedOpenAIEmbeddings batching    (line 38-71, OPENAI_EMBEDDING_BATCH_SIZE)

This probe tests each against both languages, then finds the real batch limit.

Usage:
    python probe_embedding.py
"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings

BASE = os.environ["OPENAI_BASE_URL"]
KEY = os.environ["OPENAI_API_KEY"]
MODEL = "qwen3.7-text-embedding"

CN = "2026年新能源汽车购置税减半政策将如何影响市场格局"
EN = "How will the 2026 EV purchase tax halving policy reshape the market"

DEFAULT_BATCH = int(os.environ.get("OPENAI_EMBEDDING_BATCH_SIZE", 10))


def make(check_ctx: bool, **kw):
    return OpenAIEmbeddings(
        model=MODEL,
        openai_api_base=BASE,
        openai_api_key=KEY,
        check_embedding_ctx_length=check_ctx,
        **kw,
    )


def attempt(emb, texts, label):
    try:
        vecs = emb.embed_documents(texts)
        print(f"  成功  {label:<46} -> {len(vecs)} 条向量, 维度 {len(vecs[0])}")
        return True
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        print(f"  失败  {label:<46} -> {type(exc).__name__}: {msg[:150]}")
        return False


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    print(f"模型: {MODEL}")
    print(f"端点: {BASE[:60]}...")
    print(f"OPENAI_EMBEDDING_BATCH_SIZE = {DEFAULT_BATCH}")

    # --- 实验 1: check_embedding_ctx_length 是中文问题还是语言无关? ---
    section("实验 1: check_embedding_ctx_length=True (langchain 默认) —— 按语言分别测")
    print("  这是 langchain 的默认行为：先把文本用 tiktoken 编码成 token ID 再发送\n")

    on_cn = attempt(make(True), [CN], "中文文本, 1 条, 默认设置")
    on_en = attempt(make(True), [EN], "英文文本, 1 条, 默认设置")

    section("实验 2: check_embedding_ctx_length=False (项目当前设置)")
    print("  这是 gpt_researcher 现在用的行为：直接发原始文本\n")

    off_cn = attempt(make(False), [CN], "中文文本, 1 条, 关闭预编码")
    off_en = attempt(make(False), [EN], "英文文本, 1 条, 关闭预编码")

    section("实验 1+2 结论")
    if not on_cn and not on_en:
        print("  中文和英文【都】失败 -> 这不是中文问题，是接口不兼容问题。")
        print("  任何语言的文本在默认设置下都会失败，与中文无关。")
    elif not on_cn and on_en:
        print("  只有中文失败 -> 这【确实】是中文特有的问题。")
    else:
        print(f"  默认设置下中文={'成功' if on_cn else '失败'}, "
              f"英文={'成功' if on_en else '失败'} -> 见上文原始输出。")

    # --- 实验 3: 批量上限到底是多少? ---
    section("实验 3: 批量上限实测 (check_embedding_ctx_length=False, 真实分批接口)")
    print("  BatchedOpenAIEmbeddings 声称上限 20，你的 .env 设的是 10。实测找真值。\n")

    emb = make(False)
    limit_found = None
    for n in [1, 5, 10, 15, 20, 25, 30, 32, 50, 64]:
        texts = [f"测试文本第 {i} 条，关于新能源汽车购置税政策。" for i in range(n)]
        ok = attempt(emb, texts, f"一次发送 {n} 条")
        if ok:
            limit_found = n
        else:
            print(f"         ^^ 上限在 {limit_found} 到 {n} 之间")
            break

    # --- 实验 4: 不修的话会怎样 ---
    section("实验 4: 若不做分批，一次发送大量文本会怎样?")
    print("  模拟：一次 50 条（远超大小时常出现的规模）\n")
    attempt(make(False), [f"段落 {i}: 关于购置税减免的说明。" for i in range(50)],
            "50 条不分批 (check_ctx=False)")

    # --- 总结 ---
    section("总结")
    print(f"  1. check_embedding_ctx_length=False : "
          f"{'必需' if not on_cn or not on_en else '未复现到问题'}")
    print(f"     原因: {'接口拒绝 token ID 数组，只接受原始文本' if not on_cn or not on_en else '未复现'}")
    print(f"     语言相关性: {'无 —— 中英文都会失败' if not on_cn and not on_en else '需进一步确认'}")
    print(f"  2. 分批 (batch size {DEFAULT_BATCH}) : "
          f"{'必需' if limit_found and limit_found < 50 else '在实测范围内未复现上限'}")


if __name__ == "__main__":
    main()
