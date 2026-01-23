import glob
import json
from llm import extract_scores, matching
from config import INDUSTRY_PROBLEMS
import numpy as np
import matplotlib.pyplot as plt
import asyncio

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    


def load_paper(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def plot_heatmap(all_results, paper_names=None, problem_names=None):
    plt.close("all")  
    data = np.array(all_results)

    plt.figure(figsize=(8, 6))
    plt.imshow(data)
    plt.colorbar()

    if problem_names:
        plt.xticks(range(len(problem_names)), problem_names, rotation=45, ha="right")
    else:
        plt.xlabel("Industry Problem Index")

    if paper_names:
        plt.yticks(range(len(paper_names)), paper_names)
    else:
        plt.ylabel("Paper Index")

    plt.title("Paper–Industry Problem Matching Heatmap")
    plt.tight_layout()
    plt.show()
    
    
async def process_single_paper(paper_path):
    paper = load_paper(paper_path)
    scores = [0.0] * len(INDUSTRY_PROBLEMS)

    print(f"Processing {paper_path}...")

    match_result = await matching(paper, INDUSTRY_PROBLEMS)
    matched_ids = [int(i) for i, v in match_result.items() if v]

    print(f"Matched: {matched_ids}")

    tasks = [
        extract_scores(paper, INDUSTRY_PROBLEMS[i])
        for i in matched_ids
    ]
    results = await asyncio.gather(*tasks)

    for i, r in zip(matched_ids, results):
        scores[i] = r["p_score"] * r["TRL"]
        print(
            f"[{i}] score={scores[i]} | "
            f"p_reason={r['p_score_reason']} | "
            f"TRL_reason={r['TRL_reason']}"
        )

    return scores


async def main():
    papers = glob.glob("example/*.json")

    all_results = await asyncio.gather(
        *[process_single_paper(p) for p in papers]
    )
    # plot_heatmap(
    #     all_results,
    #     paper_names=[p.split("/")[-1] for p in papers],
    #     problem_names=[f"Problem {i}" for i in range(len(INDUSTRY_PROBLEMS))]
    # )


if __name__ == "__main__":
    asyncio.run(main())