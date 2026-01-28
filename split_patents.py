#!/usr/bin/env python3
"""
拆分专利JSON文件为独立文件
将example/专利/patent_evaluation_result.json中的每个专利对象保存为单独的文件
"""

import json
import os
import re
from pathlib import Path

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除或替换非法字符"""
    # 移除或替换Windows/Linux中不允许的字符
    # 保留中文、字母、数字、空格、下划线、连字符、点号
    # 替换其他特殊字符为下划线
    # 注意：Linux允许大部分Unicode字符，但为了兼容性，我们进行清理
    illegal_chars = r'[\/:*?"<>|]'
    filename = re.sub(illegal_chars, '_', filename)
    # 移除首尾空格
    filename = filename.strip()
    # 合并连续的下划线
    filename = re.sub(r'_+', '_', filename)
    return filename

def split_patents():
    # 输入文件路径
    input_file = Path("example/专利/patent_evaluation_result.json")
    output_dir = Path("example/专利")

    if not input_file.exists():
        print(f"错误：输入文件不存在 {input_file}")
        return

    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        patents = json.load(f)

    if not isinstance(patents, list):
        print("错误：JSON根元素不是数组")
        return

    print(f"找到 {len(patents)} 个专利")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历每个专利
    for i, patent in enumerate(patents):
        # 提取标题和ID
        title = patent.get('patent_title', f'专利_{i}')
        patent_id = patent.get('patent_id', f'id_{i}')

        # 生成文件名：标题_ID.json
        # 清理标题和ID中的非法字符
        safe_title = sanitize_filename(title)
        safe_id = sanitize_filename(patent_id)

        # 如果标题过长，可以截断
        if len(safe_title) > 100:
            safe_title = safe_title[:100]

        filename = f"{safe_title}_{safe_id}.json"
        filepath = output_dir / filename

        # 避免文件名冲突（理论上不会，但以防万一）
        counter = 1
        while filepath.exists():
            filename = f"{safe_title}_{safe_id}_{counter}.json"
            filepath = output_dir / filename
            counter += 1

        # 写入单个专利文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patent, f, ensure_ascii=False, indent=2)

        print(f"已保存: {filename}")

    print(f"\n拆分完成！共保存 {len(patents)} 个专利文件到 {output_dir}")

if __name__ == "__main__":
    split_patents()