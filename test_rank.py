#!/usr/bin/env python3
"""
测试Scorer类的排名功能
"""

import asyncio
import json
from scorer import Scorer
import time

async def process_single_paper(scorer: Scorer, paper_file):
    """处理单篇论文"""
    try:
        # 处理单篇论文
        result = await scorer.process_paper(paper_file, paper_type="论文")

        if result["matched"]:
            print(f"  匹配到 {len(result['matched_problems'])} 个产业难题")
            print(f"  评分结果: {list(result['scores'].keys())}")
        else:
            print("  未匹配到任何产业难题")

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
        problems = await scorer.load_industry_problems("集成电路_0125_problems.json")
        print(f"  成功加载 {len(problems)} 个产业难题")
    except FileNotFoundError:
        print("  错误：找不到产业难题文件")
        return

    print("加载产业难题指标...")
    try:
        metrics = await scorer.load_industry_problems_metric("集成电路_0125_problems_metric.json")
        print(f"  成功加载 {len(metrics)} 个难题的指标")
    except FileNotFoundError:
        print("  警告：找不到产业难题指标文件")

    # 选择少量论文进行测试（避免过多API调用）
    import glob
    paper_dir = "example/中文文献"
    paper_files = glob.glob(f"{paper_dir}/*.json")[0:16]

    if not paper_files:
        print(f"错误：在 {paper_dir} 中找不到论文文件")
        return

    print(f"处理 {len(paper_files)} 篇论文...")

    # 并发处理所有论文
    print(f"并发处理 {len(paper_files)} 篇论文...")
    tasks = [process_single_paper(scorer, paper_file) for paper_file in paper_files]
    results = await asyncio.gather(*tasks)

    # 过滤掉失败的结果（None）
    successful_results = [r for r in results if r is not None]
    print(f"\n成功处理 {len(successful_results)} 篇论文")

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