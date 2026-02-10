#!/usr/bin/env python3
"""
整合 match_res 目录下所有 JSON 文件为一个 JSON 文件
"""

import json
import os
import glob

def merge_match_results(input_dir: str = "match_res", output_file: str = "match_res_merged.json"):
    """
    整合所有缓存结果为一个 JSON 文件

    Args:
        input_dir: 缓存文件目录
        output_file: 输出文件名
    """
    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    merged_results = []

    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                merged_results.append(data)
            except json.JSONDecodeError:
                print(f"警告: 文件格式错误，跳过 {json_file}")

    # 保存整合结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(f"整合完成: {len(merged_results)} 条结果保存到 {output_file}")
    return merged_results


if __name__ == "__main__":
    results = merge_match_results()
    print(f"\n前3条预览:")
    for i, r in enumerate(results[:3]):
        print(f"  [{i+1}] {r.get('paper_title', 'unknown')}")
