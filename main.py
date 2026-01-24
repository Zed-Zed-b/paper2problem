import glob
import json
from llm import extract_scores, matching
from config import INDUSTRY_PROBLEMS
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from asyncio import Lock

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
    

print_lock = Lock()

async def process_single_paper(paper_path):
    paper = load_paper(paper_path)
    scores = [0.0] * len(INDUSTRY_PROBLEMS)
    
    paper_name = paper_path.split("/")[-1]
    
    # 收集所有输出
    outputs = [f"\n{'='*60}"]
    outputs.append(f"📄 论文: {paper_name}")
    
    # 匹配
    match_result = await matching(paper, INDUSTRY_PROBLEMS)
    matched_ids = [int(i) for i, v in match_result.items() if v['matched']]

    # 输出所有匹配结果及理由
    outputs.append(f"📊 匹配分析:")
    for i, result in match_result.items():
        status = "✅ 匹配" if result['matched'] else "❌ 不匹配"
        outputs.append(f"  问题[{i}] {status}")
        outputs.append(f"    理由: {result['reason']}")
    
    if not matched_ids:
        outputs.append("无匹配问题，跳过")
        outputs.append(f"{'='*60}")
        async with print_lock:
            print("\n".join(outputs))
        return scores
    
    # 并发评分
    tasks = [extract_scores(paper, INDUSTRY_PROBLEMS[i]) for i in matched_ids]
    results = await asyncio.gather(*tasks)
    
    # 处理结果
    for i, r in zip(matched_ids, results):
        scores[i] = r["p_score"] * r["TRL"]
        outputs.append(f"\n  问题[{i}] - 得分: {scores[i]:.2f}")
        outputs.append(f"    P评分: {r['p_score']} - {r['p_score_reason']}...")
        outputs.append(f"    TRL: {r['TRL']} - {r['TRL_reason']}...")
    
    # 一次性输出
    async with print_lock:
        print("\n".join(outputs))
    
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