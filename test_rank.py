#!/usr/bin/env python3
"""
测试Scorer类的排名功能
"""

import asyncio
import json

from matplotlib import pyplot as plt
import numpy as np
from scorer import Scorer
import time

# 设置中文字体，添加多个后备选项
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def plot_heatmap(data:np.ndarray, 
                 paper_names=None, 
                 problem_names=None, 
                 save_path="heatmap.png"):
    """绘制并保存论文-产业难题匹配热力图"""
    plt.close("all")

    # 动态调整图形尺寸：根据数据量调整高度
    n_papers = data.shape[0]
    n_problems = data.shape[1]

    # 基础高度，每行增加一定高度
    base_height = 8
    height_per_paper = 0.2  # 每篇论文增加的高度
    fig_height = max(base_height, n_papers * height_per_paper)
    fig_width = max(12, n_problems * 0.3)  # 根据问题数量调整宽度

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=4)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_label('Level')
    cbar.ax.set_yticklabels(['0', '1', '2', '3', '4'])

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

    ax.set_title("论文-产业难题匹配热力图", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存至: {save_path}")


async def process_single_paper(scorer: Scorer, paper_file, paper_type: str = "论文"):
    """处理单篇论文"""
    try:
        # 处理单篇论文
        result = await scorer.process_paper(paper_file, paper_type=paper_type)
        print(f"\n处理{paper_type}文件: {result['paper_title']}")
        if result["matched"]:
            print(f"  匹配到 {len(result['matched_problems'])} 个产业难题")
            print(f"  匹配结果: {result['matched_problems']}")
            # print("  详细评分结果:", result['detailed_scores'])
        else:
            print("  未匹配到任何产业难题")

        with open(f'match_res/{result["paper_title"]}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


        return result
    except Exception as e:
        print(f"  处理失败: {e}")
        return None


async def main():
    """主测试函数"""
    print("初始化Scorer...")
    scorer = Scorer()

    print("加载产业难题...")
    try:
        problems = await scorer.load_industry_problems(
            "new_data/problem/checked_kpi_gen_集成电路_debate_vgemini.json",
            "new_data/problem/checked_summaries.json"
            )
        print(f"  成功加载 {len(problems)} 个产业难题")
    except FileNotFoundError:
        print("  错误：找不到产业难题文件")
        return

    # print("加载产业难题指标...")
    # try:
    #     metrics = await scorer.load_industry_problems_metric("集成电路_0125_problems_metric.json")
    #     print(f"  成功加载 {len(metrics)} 个难题的指标")
    # except FileNotFoundError:
    #     print("  警告：找不到产业难题指标文件")

    # 选择少量论文进行测试
    import glob
    paper_dir = "中文文献"
    paper_type = "论文"
    paper_files = glob.glob(f"new_data/{paper_dir}/*.json")[:10]

    if not paper_files:
        print(f"错误：在 {paper_dir} 中找不到论文文件")
        return

    print(f"处理 {len(paper_files)} 篇论文...")

    # 并发处理所有论文
    print(f"并发处理 {len(paper_files)} 篇论文...")
    tasks = [process_single_paper(scorer, paper_file, paper_type=paper_type) for paper_file in paper_files]
    results = await asyncio.gather(*tasks)

    # 过滤掉失败的结果（None）
    successful_results = [r for r in results if r is not None]
    print(f"\n成功处理 {len(successful_results)} 篇论文")

    # score_matrix, paper_titles = scorer.get_all_scores_matrix()
    # score_matrix = np.array(score_matrix, dtype = int)
    # # pass
    # plot_heatmap(score_matrix, 
    #              paper_names=paper_titles,
    #              save_path=f"{paper_dir}_heatmap.png")

    # 测试排名功能
    # print("\n" + "="*60)
    # print("测试排名功能:")
    # print("="*60)

    # # 获取所有产业难题名称
    # problem_names = scorer.get_problem_names()

    # # 测试几个有评分的难题
    # test_problems = []

    # # 查找有评分的难题
    # for paper_id in scorer.scores.keys():
    #     for problem_id in scorer.scores[paper_id].keys():
    #         if problem_id not in test_problems:
    #             test_problems.append(problem_id)

    # if not test_problems:
    #     print("无评分数据，无法测试排名功能")
    #     return

    # print(f"找到 {len(test_problems)} 个有评分的产业难题")

    # for problem_id in test_problems[:3]:  # 只测试前3个
    #     problem_name = problem_names[problem_id] if problem_id < len(problem_names) else f"问题{problem_id}"
    #     print(f"\n产业难题 {problem_id}: {problem_name}")
    #     print("-" * 80)

    #     # 方法1: 使用rank_papers_for_problem获取简单排名
    #     print("方法1: rank_papers_for_problem (简单排名)")
    #     ranked_papers = scorer.rank_papers_for_problem(problem_id)

    #     if not ranked_papers:
    #         print("  该难题暂无论文评分")
    #         continue

    #     for i, paper_info in enumerate(ranked_papers):
    #         print(f"  第{i+1}名: {paper_info['title'][:60]}...")
    #         print(f"      论文ID: {paper_info['paper_id']}")
    #         print(f"      得分信息: {paper_info['score_data'].get('total_score')}")
    #         print(f"      领域: {paper_info['domain']}")
    #         print()

        # # 方法2: 使用get_problem_rankings获取详细排名
        # print("\n方法2: get_problem_rankings (详细排名)")
        # detailed_rankings = scorer.get_problem_rankings(problem_id, include_zero_scores=False)

        # for i, ranking in enumerate(detailed_rankings[:3]):  # 只显示前3名详细信息
        #     print(f"  第{ranking['rank']}名: {ranking['title'][:60]}...")
        #     print(f"      总分: {ranking['score']:.4f}")
        #     print(f"      p_score: {ranking['p_score']}")
        #     print(f"      TRL: {ranking['TRL']}")
        #     print(f"      s_score: {ranking['s_score']:.4f}")
        #     print(f"      论文结果: {ranking['result_paper_value']}")
        #     print(f"      基线结果: {ranking['result_baseline_value']}")
        #     print()

        #     # 如果有推理信息，显示前100个字符
        #     if ranking.get('reasoning'):
        #         reasoning_preview = ranking['reasoning'][:100] + "..." if len(ranking['reasoning']) > 100 else ranking['reasoning']
        #         print(f"      推理: {reasoning_preview}")
        #         print()

        # 保存排名结果到文件
        # output_file = f"ranking_problem_{problem_id}.json"
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         "problem_id": problem_id,
        #         "problem_name": problem_name,
        #         "ranked_papers": ranked_papers,
        #         "detailed_rankings": detailed_rankings
        #     }, f, ensure_ascii=False, indent=2)
        # print(f"  排名结果已保存到: {output_file}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"\n总耗时: {end - start:.2f} 秒")