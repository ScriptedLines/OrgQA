import argparse
import asyncio
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("company_qa.eval_ragas")

DEFAULT_GOLDEN_DATASET: List[Dict[str, str]] = [
    {
        "question": "What are the core collaboration hours and how many anchor days are hybrid employees required to work in the office?",
        "ground_truth": "Core collaboration hours are 10:00 AM to 3:00 PM in the team's designated primary time zone. Hybrid employees situated within a 40-mile radius of an office hub must work in-office a minimum of two designated anchor days per week (ordinarily Tuesdays and Thursdays).",
    },
    {
        "question": "How much is the one-time ergonomic home office grant and the monthly internet reimbursement?",
        "ground_truth": "Eligible hybrid and fully remote full-time employees receive a one-time ergonomic home office setup grant of $1,200 USD disbursed after thirty days of employment, and a monthly non-taxable internet and connectivity reimbursement of $85 USD.",
    },
    {
        "question": "Can employees work remotely from another country, for how long, and how far in advance must they apply?",
        "ground_truth": "Employees in good standing may request authorization to work remotely from an approved international jurisdiction for up to thirty calendar days per rolling 12-month period. Requests must be submitted via the Global Mobility Portal at least twenty-one calendar days prior to departure.",
    },
    {
        "question": "What is the policy regarding working outside the country in OFAC-sanctioned jurisdictions?",
        "ground_truth": "Due to trade embargoes, data residency sanctions, and international tax treaties, remote work is categorically prohibited in OFAC-sanctioned regions, countries under US State Department Level 4 Travel Advisories, and jurisdictions lacking dual-taxation protection.",
    },
    {
        "question": "What are the minimum internet connection speed requirements to qualify for remote work?",
        "ground_truth": "Employees must maintain high-speed broadband internet delivering minimum sustained speeds of 100 Mbps download and 20 Mbps upload with latency under 40ms.",
    },
    {
        "question": "What is the coworking space membership allowance for remote employees?",
        "ground_truth": "Upon manager approval, the Company reimburses up to $300 USD per month for partner coworking facilities (e.g., WeWork, Spaces, Regus) for open-desk passes. Private dedicated office upgrades are not reimbursable.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Production Ragas Evaluation Pipeline for Enterprise RAG")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="benchmark_dataset.json" if Path("benchmark_dataset.json").exists() else None,
        help="Path to an external JSON dataset file (list of objects with 'question' and 'ground_truth').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation reports (CSV, JSON, Markdown).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum concurrent RAG queries to prevent rate-limiting.",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.80,
        help="Minimum required average faithfulness score to pass CI/CD gate.",
    )
    parser.add_argument(
        "--min-relevancy",
        type=float,
        default=0.80,
        help="Minimum required average answer relevancy score to pass CI/CD gate.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.30,
        help="Similarity score threshold for base vector retriever (default: 0.30 for NVIDIA nemotron-3-embed-1b).",
    )
    return parser.parse_args()


async def _process_single_sample(
    sample: Dict[str, str],
    semaphore: asyncio.Semaphore,
    rag_query_chain,
    qa_chain,
    retriever,
    max_retries: int = 3,
) -> Tuple[str, str, List[str], str]:
    query = sample["question"]
    ground_truth = sample["ground_truth"]

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Processing sample: '{query[:45]}...' (Attempt {attempt})")
                optimized_query = await rag_query_chain.ainvoke(
                    {"summary": "", "messages": [], "query": query}
                )
                relevant_chunks = await retriever.ainvoke(optimized_query)

                if not relevant_chunks:
                    answer = (
                        "I don't have enough information in the available policy documents "
                        "to answer that. Please contact HR or IT for assistance."
                    )
                    chunk_texts = ["No relevant context found."]
                else:
                    context_str = "\n\n".join(
                        f"[{i + 1}] Source: {d.metadata.get('source')}\n{d.page_content}"
                        for i, d in enumerate(relevant_chunks)
                    )
                    answer = await qa_chain.ainvoke(
                        {
                            "query": query,
                            "summary": "",
                            "recent_history": [],
                            "context": context_str,
                            "user_profile": "Role: employee, Department: General, Location: Global",
                        }
                    )
                    chunk_texts = [d.page_content for d in relevant_chunks]

                return query, answer, chunk_texts, ground_truth

            except Exception as e:
                logger.warning(f"Error on attempt {attempt} for '{query[:40]}': {e}")
                if attempt == max_retries:
                    logger.error(f"Failed to evaluate sample after {max_retries} attempts: {query}")
                    return query, f"Error: {str(e)}", ["Error retrieving context."], ground_truth
                await asyncio.sleep(2 ** attempt)


async def collect_rag_samples(
    test_cases: List[Dict[str, str]],
    llm,
    reranker,
    vector_store,
    rag_query_generator_prompt,
    qa_prompt,
    concurrency: int = 2,
    score_threshold: float = 0.30,
):
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_core.output_parsers import StrOutputParser
    from datasets import Dataset

    semaphore = asyncio.Semaphore(concurrency)
    rag_query_chain = rag_query_generator_prompt | llm | StrOutputParser()
    qa_chain = qa_prompt | llm | StrOutputParser()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 50,
                "score_threshold": score_threshold,
                "filter": {"status": "active"},
            }
        ),
    )

    tasks = [
        _process_single_sample(tc, semaphore, rag_query_chain, qa_chain, compression_retriever)
        for tc in test_cases
    ]

    results = await asyncio.gather(*tasks)
    questions, answers, contexts, ground_truths = zip(*results)

    return Dataset.from_dict({
        "question": list(questions),
        "answer": list(answers),
        "contexts": list(contexts),
        "ground_truth": list(ground_truths),
    })


def generate_markdown_report(df, aggregate_scores: Dict[str, float], output_path: Path):
    md_content = [
        "# 📊 RAG Benchmark Evaluation Report (Ragas)",
        f"\n**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Samples Evaluated:** {len(df)}  \n",
        "## 📈 Summary Metric Scores\n",
        "| Metric | Score | Target Threshold | Status |",
        "| :--- | :---: | :---: | :---: |",
    ]

    targets = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.70,
    }

    for metric, score in aggregate_scores.items():
        target = targets.get(metric, 0.75)
        status = "✅ PASS" if score >= target else "❌ FAIL"
        md_content.append(f"| `{metric}` | **{score:.4f}** | {target:.2f} | {status} |")

    md_content.append("\n## 🔍 Per-Sample Detailed Breakdown\n")
    md_content.append("| # | Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |")
    md_content.append("| :-: | :--- | :-: | :-: | :-: | :-: |")

    for idx, row in df.iterrows():
        raw_q = str(row.get("question") or row.get("user_input") or f"Sample {idx + 1}")
        q_short = raw_q[:60] + "..." if len(raw_q) > 60 else raw_q
        md_content.append(
            f"| {idx + 1} | {q_short} | "
            f"{row.get('faithfulness', 0.0):.2f} | "
            f"{row.get('answer_relevancy', 0.0):.2f} | "
            f"{row.get('context_precision', 0.0):.2f} | "
            f"{row.get('context_recall', 0.0):.2f} |"
        )

    output_path.write_text("\n".join(md_content), encoding="utf-8")
    logger.info(f"Markdown report generated at: {output_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        _m = types.ModuleType("langchain_community.chat_models.vertexai")
        _m.ChatVertexAI = None
        sys.modules["langchain_community.chat_models.vertexai"] = _m

    from config import embeddings, llm, reranker, vector_store
    from prompt import qa_prompt, rag_query_generator_prompt
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    if args.dataset_path:
        dataset_file = Path(args.dataset_path)
        if not dataset_file.exists():
            logger.error(f"Dataset file not found at: {dataset_file}")
            sys.exit(1)
        test_cases = json.loads(dataset_file.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(test_cases)} samples from {dataset_file}")
    else:
        test_cases = DEFAULT_GOLDEN_DATASET
        logger.info(f"Using default golden benchmark ({len(test_cases)} samples).")

    logger.info(f"Collecting RAG responses (Concurrency: {args.concurrency})...")
    ragas_dataset = asyncio.run(
        collect_rag_samples(
            test_cases=test_cases,
            llm=llm,
            reranker=reranker,
            vector_store=vector_store,
            rag_query_generator_prompt=rag_query_generator_prompt,
            qa_prompt=qa_prompt,
            concurrency=args.concurrency,
            score_threshold=args.score_threshold,
        )
    )

    logger.info("Initializing Ragas evaluators with Gemini LLM and NVIDIA Embeddings...")
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    logger.info("Scoring metrics with Ragas...")
    eval_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    df = eval_result.to_pandas()
    aggregate_scores = {}
    if hasattr(eval_result, "_repr_dict") and eval_result._repr_dict:
        aggregate_scores = {k: float(v) for k, v in eval_result._repr_dict.items()}
    else:
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in df.columns:
                aggregate_scores[col] = float(df[col].mean())

    csv_path = output_dir / "ragas_results.csv"
    json_path = output_dir / "ragas_metrics.json"
    md_path = output_dir / "ragas_report.md"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(aggregate_scores, indent=2), encoding="utf-8")
    generate_markdown_report(df, aggregate_scores, md_path)

    print("\n" + "=" * 80)
    print("📊 AGGREGATE RAGAS SCORES")
    print("=" * 80)
    for metric, score in aggregate_scores.items():
        print(f"  • {metric:20s}: {score:.4f}")
    print("=" * 80)

    failed_metrics = []
    if aggregate_scores.get("faithfulness", 0.0) < args.min_faithfulness:
        failed_metrics.append(f"Faithfulness ({aggregate_scores.get('faithfulness', 0):.2f} < {args.min_faithfulness})")
    if aggregate_scores.get("answer_relevancy", 0.0) < args.min_relevancy:
        failed_metrics.append(f"Answer Relevancy ({aggregate_scores.get('answer_relevancy', 0):.2f} < {args.min_relevancy})")

    if failed_metrics:
        logger.error(f"❌ CI/CD Quality Gate FAILED on: {', '.join(failed_metrics)}")
        sys.exit(1)
    else:
        logger.info("✅ CI/CD Quality Gate PASSED successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
