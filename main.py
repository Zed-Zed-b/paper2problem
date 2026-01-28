import glob
import json
import os
from llm import extract_scores, matching, decide_metric
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

def load_industry_problems_metric():
    json_data = None
    with open(r"集成电路_0125_problems_metric.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    res = json_data["results"]
    # problems_ids = [i.pop("problem_id", None) - 1 for i in res]
    metrics = {_item["problem_id"] - 1: _item["metrics"] for _item in res}
    # for problem in problems:
    #     problem.pop("id", None)

    return metrics

def load_industry_problems():
    json_data = None
    with open(r"集成电路_0125_problems.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    sub_fields = json_data["sub_fields"]
    problems = []
    for item in sub_fields:
        problems.extend(item["problems"])

    for problem in problems:
        problem.pop("id", None)

    return problems

INDUSTRY_PROBLEMS = load_industry_problems()
INDUSTRY_PROBLEMS_METRIC = load_industry_problems_metric()

def load_paper(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def plot_heatmap(all_results, paper_names=None, problem_names=None, save_path="heatmap.png"):
    """绘制并保存论文-产业难题匹配热力图"""
    plt.close("all")
    data = np.array(all_results)

    # 动态调整图形尺寸：根据数据量调整高度
    n_papers = data.shape[0]
    n_problems = data.shape[1]

    # 基础高度，每行增加一定高度
    base_height = 8
    height_per_paper = 0.2  # 每篇论文增加的高度
    fig_height = max(base_height, n_papers * height_per_paper)
    fig_width = max(12, n_problems * 0.3)  # 根据问题数量调整宽度

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('得分')

    # 设置横轴标签（产业难题）
    if problem_names:
        # 如果问题数量过多，间隔显示标签
        if len(problem_names) > 30:
            step = max(1, len(problem_names) // 20)  # 最多显示20个标签
            xticks = range(0, len(problem_names), step)
            xticklabels = [problem_names[i] for i in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticks(range(len(problem_names)))
            ax.set_xticklabels(problem_names, rotation=45, ha='right', fontsize=9)
    else:
        ax.set_xticks(range(data.shape[1]))
        ax.set_xlabel("产业难题编号", fontsize=10)

    # 设置纵轴标签（论文名称）
    if paper_names:
        # 如果论文数量过多，调整显示策略
        if len(paper_names) > 30:
            # 间隔显示标签
            step = max(1, len(paper_names) // 30)  # 最多显示30个标签
            yticks = range(0, len(paper_names), step)
            yticklabels = [paper_names[i] for i in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=7, rotation=0)

            # 添加网格线帮助定位
            ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
        else:
            ax.set_yticks(range(len(paper_names)))
            # 如果论文名称过长，进行截断
            truncated_names = []
            max_len = 50  # 最大显示长度
            for name in paper_names:
                if len(name) > max_len:
                    truncated_names.append(name[:max_len] + "...")
                else:
                    truncated_names.append(name)
            ax.set_yticklabels(truncated_names, fontsize=8, rotation=0)
    else:
        ax.set_yticks(range(data.shape[0]))
        ax.set_ylabel("论文编号", fontsize=10)

    # 在每个格子中显示数值（如果数据量不大）
    if n_papers <= 20 and n_problems <= 20:
        for i in range(len(all_results)):
            for j in range(len(all_results[i])):
                if all_results[i][j] > 0:  # 只显示非零值
                    text = ax.text(j, i, f'{all_results[i][j]:.1f}',
                                ha="center", va="center", color="black", fontsize=7)

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

    try:
        metric_tasks = []
        for i in matched_ids:
            if i in INDUSTRY_PROBLEMS_METRIC:
                metric_tasks.append(decide_metric(paper, INDUSTRY_PROBLEMS_METRIC[i], paper_type=paper_type))
        if metric_tasks:
            metric_results = await asyncio.gather(*metric_tasks)
            os.makedirs("metric_match", exist_ok=True)
            with open(f"metric_match/{paper_name}.json", "w", encoding="utf-8") as f:
                json.dump(metric_results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        outputs.append(f"评分过程中出现错误: {e}")
        outputs.append(f"模型输出: {match_result}")
        async with print_lock:
            print("\n".join(outputs))
        raise e
    
    
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
    paper_dir = "中文文献"
    papers = glob.glob(f"example/{paper_dir}/*.json")

    all_results = await asyncio.gather(
        *[process_single_paper(p, paper_type=paper_type) for p in papers]
    )

    # 生成热力图
    plot_heatmap(
        all_results,
        paper_names=[p.split("/")[-1].replace(".json", "") for p in papers],
        problem_names=None,
        save_path=f"{paper_dir}_heatmap.png"
    )


if __name__ == "__main__":
    asyncio.run(main())
