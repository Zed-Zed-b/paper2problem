import glob
import json
from llm import extract_scores, matching
# from config import INDUSTRY_PROBLEMS
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import asyncio
from asyncio import Lock

# 设置中文字体，添加多个后备选项
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def load_industry_problems():
    json_data = None
    with open(r"集成电路_0125.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    sub_fields = json_data["sub_fields"]
    problems = []
    for item in sub_fields:
        problems.extend(item["problems"])

    for problem in problems:
        problem.pop("id", None)

    return problems

INDUSTRY_PROBLEMS = load_industry_problems()

def load_paper(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def plot_heatmap(all_results, paper_names=None, problem_names=None, save_path="heatmap.png"):
    """绘制并保存论文-产业难题匹配热力图"""
    plt.close("all")
    data = np.array(all_results)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('得分 (p_score × TRL)')

    # 设置横轴标签（产业难题）
    if problem_names:
        ax.set_xticks(range(len(problem_names)))
        ax.set_xticklabels(problem_names, rotation=45, ha='right', fontsize=9)
    else:
        ax.set_xticks(range(data.shape[1]))
        ax.set_xlabel("产业难题编号", fontsize=10)

    # 设置纵轴标签（论文名称）
    if paper_names:
        ax.set_yticks(range(len(paper_names)))
        ax.set_yticklabels(paper_names, fontsize=9)
    else:
        ax.set_ylabel("论文编号", fontsize=10)

    # 在每个格子中显示数值
    # for i in range(len(all_results)):
    #     for j in range(len(all_results[i])):
    #         text = ax.text(j, i, f'{all_results[i][j]:.1f}',
    #                     ha="center", va="center", color="black", fontsize=8)

    ax.set_title("论文-产业难题匹配热力图", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存至: {save_path}")
    

print_lock = Lock()

async def process_single_paper(paper_path, paper_type="论文"):
    paper = load_paper(paper_path)
    scores = [0.0] * len(INDUSTRY_PROBLEMS)
    
    paper_name = paper_path.split("/")[-1]
    
    # 收集所有输出
    outputs = [f"\n{'='*60}"]
    outputs.append(f"📄 {paper_type}: {paper_name}")
    
    # 匹配
    match_result = await matching(paper, INDUSTRY_PROBLEMS, paper_type=paper_type)
    matched_ids = [int(i) for i, v in match_result.items() if v['matched']]

    # 输出所有匹配结果及理由
    # outputs.append(f"📊 匹配分析:")
    for i, result in match_result.items():
        if result['matched']:
            status = "✅ 匹配"
            outputs.append(f"  问题[{i}] {status}")
            outputs.append(f"    理由: {result['reason']}")
    
    if not matched_ids:
        outputs.append("无匹配问题，跳过")
        outputs.append(f"{'='*60}")
        async with print_lock:
            print("\n".join(outputs))
        return scores
    
    # 并发评分
    try:
        tasks = [extract_scores(paper, INDUSTRY_PROBLEMS[i], paper_type=paper_type) for i in matched_ids]
        results = await asyncio.gather(*tasks)
    except Exception as e:
        outputs.append(f"评分过程中出现错误: {e}")
        outputs.append(f"模型输出: {match_result}")
        async with print_lock:
            print("\n".join(outputs))
        raise e
        # return scores
    
    # 处理结果
    for i, r in zip(matched_ids, results):
        rp = eval(r["result_paper"]) if isinstance(r["result_paper"], str) else r["result_paper"]
        rb = eval(r["result_baseline"]) if isinstance(r["result_baseline"], str) else r["result_baseline"]
        s_score = math.tanh(math.fabs((rp - rb) / rb)) if rb != 0 else 0.0
        scores[i] = r["p_score"] * r["TRL"] * (1 + s_score)
        outputs.append(f"\n  问题[{i}] - 得分: {scores[i]:.2f}")
        outputs.append(f"    P评分: {r['p_score']} - {r['p_score_reason']}")
        outputs.append(f"    TRL: {r['TRL']} - {r['TRL_reason']}")
        outputs.append(f"    result_paper: {rp}, result_baseline: {rb} - s_score: {s_score:.4f} - {r['s_score_reason']}")
    
    # 一次性输出
    async with print_lock:
        print("\n".join(outputs))
    return scores


async def main():
    paper_type = "论文"
    papers = glob.glob("example/*.json")

    all_results = await asyncio.gather(
        *[process_single_paper(p, paper_type=paper_type) for p in papers]
    )

    # 生成热力图
    plot_heatmap(
        all_results,
        paper_names=[p.split("/")[-1].replace(".json", "") for p in papers],
        problem_names=None,
        save_path="heatmap.png"
    )


if __name__ == "__main__":
    asyncio.run(main())
