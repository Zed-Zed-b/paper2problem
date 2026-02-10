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
    """处理单篇论文，返回 (状态, paper_title, error_msg)"""
    try:
        # 处理单篇论文
        result = await scorer.process_paper(paper_file, 
                                            paper_type=paper_type,
                                            use_cache=False)

        # 保存时移除 _from_cache 标志
        result_to_save = {k: v for k, v in result.items() if k != "_from_cache"}
        with open(f'match_res/{result["paper_title"].replace("/", "_")}.json', 'w', encoding='utf-8') as f:
            json.dump(result_to_save, f, ensure_ascii=False, indent=4)

        # 返回结果类型：是否从缓存读取
        return ("cached", result["paper_title"]) if result.get("_from_cache") else ("new", result["paper_title"])
    except Exception as e:
        return ("error", paper_file, str(e))


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

    # 选择论文进行测试
    import glob
    paper_dir = "中文文献"
    paper_type = "论文"
    paper_files = glob.glob(f"new_data/{paper_dir}/*.json")

    if not paper_files:
        print(f"错误：在 {paper_dir} 中找不到论文文件")
        return

    # 分批处理配置
    BATCH_SIZE = 20  # 每批处理的论文数量

    n_batches = (len(paper_files) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"共 {len(paper_files)} 篇论文，分为 {n_batches} 批处理 (每批 {BATCH_SIZE} 篇)")

    all_results = []

    # 逐批串行处理
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, len(paper_files))
        batch_files = paper_files[start:end]

        print(f"\n--- 处理第 {batch_idx + 1}/{n_batches} 批 ({end - start} 篇) ---")

        # batch 内并发处理
        tasks = [process_single_paper(scorer, paper_file, paper_type=paper_type) for paper_file in batch_files]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

    # 汇总统计
    cached_count = sum(1 for r in all_results if r[0] == "cached")
    new_count = sum(1 for r in all_results if r[0] == "new")
    error_count = sum(1 for r in all_results if r[0] == "error")
    error_files = [r for r in all_results if r[0] == "error"]

    print(f"\n处理完成:")
    print(f"  缓存复用: {cached_count} 篇")
    print(f"  新增处理: {new_count} 篇")
    print(f"  处理失败: {error_count} 篇")

    # 打印处理失败的文件
    if error_files:
        print(f"\n处理失败的文件:")
        for status, file_path, error_msg in error_files:
            print(f"  - {file_path}")
            print(f"    错误: {error_msg}")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"\n总耗时: {end - start:.2f} 秒")