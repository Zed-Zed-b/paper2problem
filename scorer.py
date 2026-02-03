#!/usr/bin/env python3
"""
Scorer类：整合LLM评分和匹配功能
将llm.py和main.py的功能重构为面向对象的设计
"""

import json
import os
import glob
import asyncio
import math
from typing import Dict, List, Any, Optional
from openai import OpenAI
import re
from paper_and_problem import Paper, IndustryProblem


class Scorer:
    def __init__(self, model: str = "deepseek-chat", 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None):
        """
        初始化Scorer

        Args:
            model: 模型名称
            api_key: OpenAI API密钥，如果为None则使用默认值
            base_url: API基础URL，如果为None则使用默认值
        """
        self.model = model
        self.api_key = api_key or "sk-703b713308054f34b710539c87788d82"
        self.base_url = base_url or "https://api.deepseek.com"

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 加载prompts
        self.prompts = self._load_prompts()

        # 存储产业难题和指标
        self.industry_problems: List[Dict[str, Any]] = []
        self.industry_problems_metric: Dict[int, List[Dict[str, Any]]] = {}

        # 存储Paper和IndustryProblem对象
        self.papers: Dict[str, Paper] = {}  # key: paper_id or file_path
        self.problems: Dict[int, IndustryProblem] = {}  # key: problem_id

        # 存储评分结果
        self.scores: Dict[str, Dict[int, Dict[str, Any]]] = {}  # paper_id -> problem_id -> score_data
        self.metric_results: Dict[str, Dict[int, Dict[str, Any]]] = {}  # paper_id -> problem_id -> metric_results
        self.match_results: Dict[str, Dict[int, bool]] = {}  # paper_id -> problem_id -> matched

    def _load_prompts(self) -> Dict[str, str]:
        """加载所有prompt文件"""
        prompts_dir = "prompts"
        prompts = {}

        prompt_files = {
            "match_paper": "match_papers.txt",
            "match_patent": "match_patents.txt",
            "score_paper": "score_papers.txt",
            "score_patent": "score_patents.txt",
            "decide_metric": "decide_metric.txt"
        }

        for key, filename in prompt_files.items():
            filepath = os.path.join(prompts_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read()
            except FileNotFoundError:
                print(f"警告: 找不到prompt文件 {filepath}")
                prompts[key] = ""

        return prompts

    def safe_json_loads(self, llm_text: str) -> Dict[str, Any]:
        """安全解析LLM返回的JSON，处理常见的格式问题"""
        llm_text = llm_text.strip()

        # 提取 ```json``` 代码块
        if "```" in llm_text:
            # 找到第一个 ``` 和最后一个 ```
            start = llm_text.find("```")
            end = llm_text.rfind("```")
            if start != -1 and end != -1:
                # 获取代码块内容
                code_block = llm_text[start:end]
                # 移除 ```json 或 ``` 标记
                code_block = re.sub(r"^```(?:json)?\s*", "", code_block)
                llm_text = code_block.strip()

        # 尝试直接解析
        try:
            return json.loads(llm_text)
        except json.decoder.JSONDecodeError:
            # 如果失败，尝试提取第一个完整的JSON对象
            # 从第一个 { 开始匹配
            start = llm_text.find("{")
            if start == -1:
                raise

            # 计算括号匹配，找到完整的JSON对象
            brace_count = 0
            in_string = False
            escape_next = False
            end = -1

            for i in range(start, len(llm_text)):
                char = llm_text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\" and in_string:
                    escape_next = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break

            if end == -1:
                raise

            return json.loads(llm_text[start:end])

    def load_industry_problems(self, problems_file: str = "集成电路_0125_problems.json"):
        """加载产业难题"""
        with open(problems_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        sub_fields = json_data["sub_fields"]
        problems = []
        for item in sub_fields:
            problems.extend(item["problems"])

        # 移除id字段，避免混淆
        for problem in problems:
            problem.pop("id", None)

        self.industry_problems = problems

        # 创建IndustryProblem对象
        for i, problem_data in enumerate(problems):
            problem = IndustryProblem(id=str(i), metadata=problem_data)
            self.problems[i] = problem

        return problems

    def load_industry_problems_metric(self, metric_file: str = "集成电路_0125_problems_metric.json"):
        """加载产业难题指标"""
        with open(metric_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        res = json_data["results"]
        metrics = {_item["problem_id"] - 1: _item["metrics"] for _item in res}
        self.industry_problems_metric = metrics
        return metrics

    def load_paper(self, paper_path: str, paper_type: str = "论文") -> Paper:
        """加载单篇论文并创建Paper对象

        Args:
            paper_path: 论文文件路径
            paper_type: 论文类型，可选"论文"或"专利"，默认为"论文"
        """
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)

        # 从论文数据中提取标题和领域
        title = paper_data.get('title',
                               paper_data.get('paper_title',
                                              paper_data.get('patent_title', '未知标题')))
        domain = paper_data.get('domain', '未知领域')

        # 创建Paper对象
        metadata = {
            'title': title,
            'domain': domain,
            'raw_data': paper_data
        }

        paper = Paper(metadata, paper_type=paper_type)
        paper.file_path = paper_path
        paper.paper_id = os.path.basename(paper_path).replace('.json', '')

        # 存储到papers字典
        self.papers[paper.paper_id] = paper

        return paper

    def load_papers_from_directory(self, directory: str, paper_type: str = "论文") -> List[Paper]:
        """从目录加载所有论文

        Args:
            directory: 目录路径
            paper_type: 论文类型，可选"论文"或"专利"，默认为"论文"
        """
        paper_files = glob.glob(os.path.join(directory, "*.json"))
        papers = []

        for paper_file in paper_files:
            paper = self.load_paper(paper_file, paper_type=paper_type)
            papers.append(paper)

        return papers

    async def match_paper_to_problems(self, paper: Paper) -> Dict[int, bool]:
        """
        匹配论文与产业难题

        Args:
            paper: Paper对象
            paper_type: 论文类型，可选"论文"或"专利"

        Returns:
            匹配结果字典，problem_id -> matched (bool)
        """
        # 选择正确的prompt
        paper_type = paper.paper_type
        if paper.paper_type == "专利":
            prompt_template = self.prompts.get("match_patent", "")
        else:
            prompt_template = self.prompts.get("match_paper", "")

        if not prompt_template:
            raise ValueError(f"找不到{paper_type}匹配的prompt模板")

        # 准备产业难题字典
        industry_problems_dict = {i: prob for i, prob in enumerate(self.industry_problems)}

        # 构建完整prompt
        prompt = prompt_template
        # 替换{paper_type}占位符
        prompt = prompt.replace("{paper_type}", paper_type)
        # 添加论文内容
        paper_content = json.dumps(paper.metadata['raw_data'], ensure_ascii=False, indent=2)
        prompt += f"\n\n【{paper_type}内容】\n{paper_content}"
        # 添加产业难题
        prompt += f"\n\n【产业难题】\n{industry_problems_dict}"

        # 调用LLM
        resp = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # 解析结果
        result = self.safe_json_loads(resp.choices[0].message.content)

        # 转换为problem_id -> bool的字典
        match_results = {}
        for key, value in result.items():
            try:
                problem_id = int(key)
                matched = value.get('matched', False)
                match_results[problem_id] = matched

                # 如果匹配，更新Paper和IndustryProblem对象
                if matched:
                    paper.add_problem(str(problem_id))
                    if problem_id in self.problems:
                        self.problems[problem_id].add_paper(paper)

                # 存储匹配结果
                if paper.paper_id not in self.match_results:
                    self.match_results[paper.paper_id] = {}
                self.match_results[paper.paper_id][problem_id] = matched

            except (ValueError, KeyError):
                continue

        return match_results

    async def score_paper_for_problem(self, paper: Paper, problem_id: int) -> Dict[str, Any]:
        """
        对论文在特定产业难题上进行评分

        Args:
            paper: Paper对象
            problem_id: 产业难题ID
            paper_type: 论文类型

        Returns:
            评分结果字典
        """
        paper_type = paper.paper_type
        if problem_id >= len(self.industry_problems):
            raise ValueError(f"无效的problem_id: {problem_id}")

        # 选择正确的prompt
        if paper_type == "专利":
            prompt_template = self.prompts.get("score_patent", "")
        else:
            prompt_template = self.prompts.get("score_paper", "")

        if not prompt_template:
            raise ValueError(f"找不到{paper_type}评分的prompt模板")

        # 获取产业难题
        industry_problem = self.industry_problems[problem_id]

        # 构建完整prompt
        prompt = prompt_template
        # 替换{paper_type}占位符
        prompt = prompt.replace("{paper_type}", paper_type)
        # 添加论文内容和产业难题
        paper_content = json.dumps(paper.metadata['raw_data'], ensure_ascii=False, indent=2)
        prompt += f"\n\n{paper_type}内容:\n{paper_content}"
        prompt += f"\n\n产业难题:\n{json.dumps(industry_problem, ensure_ascii=False, indent=2)}"

        # 调用LLM
        resp = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # 解析结果
        score_result = self.safe_json_loads(resp.choices[0].message.content)

        # 计算s_score
        rp = eval(score_result["result_paper"]) if isinstance(score_result["result_paper"], str) else score_result["result_paper"]
        rb = eval(score_result["result_baseline"]) if isinstance(score_result["result_baseline"], str) else score_result["result_baseline"]
        s_score = math.tanh(math.fabs((rp - rb) / rb)) if rb != 0 else 0.0

        # 计算总得分
        p_score = score_result.get("p_score", 0)
        trl = score_result.get("TRL", 0)
        total_score = p_score * trl * (1 + s_score)

        # 构建完整结果
        result = {
            **score_result,
            "result_paper_value": rp,
            "result_baseline_value": rb,
            "s_score": s_score,
            "total_score": total_score,
            "problem_id": problem_id
        }

        # 存储结果
        if paper.paper_id not in self.scores:
            self.scores[paper.paper_id] = {}
        self.scores[paper.paper_id][problem_id] = result

        return result

    async def evaluate_metrics_for_paper(self, paper: Paper, problem_id: int, paper_type: str = "论文") -> Dict[str, Any]:
        """
        评估论文在特定产业难题上的指标

        Args:
            paper: Paper对象
            problem_id: 产业难题ID
            paper_type: 论文类型

        Returns:
            指标评估结果
        """
        if problem_id not in self.industry_problems_metric:
            return {"problem_id": problem_id, "metrics": {}, "has_metrics": False}

        metric_list = self.industry_problems_metric[problem_id]

        # 获取decide_metric prompt
        prompt_template = self.prompts.get("decide_metric", "")
        if not prompt_template:
            raise ValueError("找不到指标评估的prompt模板")

        # 构建完整prompt
        prompt = prompt_template
        # 替换{paper_type}占位符
        prompt = prompt.replace("{paper_type}", paper_type)
        # 添加论文内容和指标列表
        paper_content = json.dumps(paper.metadata['raw_data'], ensure_ascii=False, indent=2)
        prompt += f"\n\n{paper_type}内容:\n{paper_content}"
        prompt += f"\n\nmetric列表:\n{json.dumps(metric_list, ensure_ascii=False, indent=2)}"

        # 调用LLM
        resp = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # 解析结果
        metric_result = self.safe_json_loads(resp.choices[0].message.content)

        # 构建结果
        result = {
            "problem_id": problem_id,
            "metrics": metric_result,
            "has_metrics": any(metric.get("provided", False) for metric in metric_result.values())
        }

        # 存储结果
        if paper.paper_id not in self.metric_results:
            self.metric_results[paper.paper_id] = {}
        self.metric_results[paper.paper_id][problem_id] = result

        return result

    async def process_paper(self, paper_path: str, paper_type: str = "论文") -> Dict[str, Any]:
        """
        处理单篇论文的完整流程

        Args:
            paper_path: 论文文件路径
            paper_type: 论文类型

        Returns:
            处理结果
        """
        # 加载论文
        paper = self.load_paper(paper_path, paper_type=paper_type)

        # 匹配论文与产业难题
        match_results = await self.match_paper_to_problems(paper, paper_type)
        matched_ids = [pid for pid, matched in match_results.items() if matched]

        if not matched_ids:
            return {
                "paper_id": paper.paper_id,
                "matched": False,
                "scores": {},
                "metric_results": {}
            }

        # 并发评分
        score_tasks = [self.score_paper_for_problem(paper, pid, paper_type) for pid in matched_ids]
        score_results = await asyncio.gather(*score_tasks)

        # 并发评估指标
        metric_tasks = []
        for pid in matched_ids:
            if pid in self.industry_problems_metric:
                metric_tasks.append(self.evaluate_metrics_for_paper(paper, pid, paper_type))

        metric_results = []
        if metric_tasks:
            metric_results = await asyncio.gather(*metric_tasks)

        # 保存指标结果到文件
        if metric_results:
            os.makedirs("metric_match", exist_ok=True)
            output_file = f"metric_match/{paper.paper_id}.json"

            # 重新组织数据结构
            reorganized_results = []
            for result in metric_results:
                reorganized_results.append({
                    "problem_id": result["problem_id"] + 1,
                    "metrics": result["metrics"]
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(reorganized_results, f, ensure_ascii=False, indent=4)

        # 构建最终结果
        result = {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "matched": True,
            "matched_problems": matched_ids,
            "scores": {r["problem_id"]: r for r in score_results},
            "metric_results": {r["problem_id"]: r for r in metric_results} if metric_results else {}
        }

        return result

    def get_paper_scores(self, paper_id: str) -> List[float]:
        """
        获取论文在所有产业难题上的得分向量

        Returns:
            得分列表，长度为产业难题数量，未匹配的问题得分为0
        """
        if not self.industry_problems:
            return []

        scores = [0.0] * len(self.industry_problems)

        if paper_id in self.scores:
            for problem_id, score_data in self.scores[paper_id].items():
                if problem_id < len(scores):
                    scores[problem_id] = score_data.get("total_score", 0.0)

        return scores

    def get_all_scores_matrix(self) -> List[List[float]]:
        """
        获取所有论文的得分矩阵

        Returns:
            得分矩阵，每行代表一篇论文，每列代表一个产业难题
        """
        if not self.papers:
            return []

        matrix = []
        for paper_id in self.papers.keys():
            scores = self.get_paper_scores(paper_id)
            matrix.append(scores)

        return matrix

    def get_paper_names(self) -> List[str]:
        """获取所有论文的名称"""
        return [paper.title for paper in self.papers.values()]

    def get_problem_names(self) -> List[str]:
        """获取所有产业难题的名称"""
        names = []
        for i, problem in enumerate(self.industry_problems):
            name = problem.get('title', f'问题{i}')
            names.append(name)
        return names

    def rank_papers_for_problem(self, problem_id: int, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        对特定产业难题的论文进行排名（从高到低）

        Args:
            problem_id: 产业难题ID
            top_n: 返回前N名，如果为None则返回所有

        Returns:
            排序后的论文列表，每个元素包含paper_id、title、score等信息
        """
        if problem_id >= len(self.industry_problems):
            raise ValueError(f"无效的problem_id: {problem_id}")

        ranked_papers = []

        # 遍历所有论文
        for paper_id, paper in self.papers.items():
            score = 0.0
            score_data = None

            # 检查是否有评分记录
            if paper_id in self.scores and problem_id in self.scores[paper_id]:
                score_data = self.scores[paper_id][problem_id]
                score = score_data.get("total_score", 0.0)
            else:
                # 检查是否有匹配记录但未评分的情况
                if paper_id in self.match_results and self.match_results[paper_id].get(problem_id, False):
                    # 匹配但未评分，得分为0
                    score = 0.0
                else:
                    # 未匹配也未评分，跳过
                    continue

            # 构建论文信息
            paper_info = {
                "paper_id": paper_id,
                "title": paper.title,
                "domain": paper.domain,
                "score": score,
                "score_data": score_data,
                "matched": score_data is not None or (paper_id in self.match_results and self.match_results[paper_id].get(problem_id, False))
            }

            ranked_papers.append(paper_info)

        # 按分数从高到低排序
        ranked_papers.sort(key=lambda x: x["score"], reverse=True)

        # 限制返回数量
        if top_n is not None and top_n > 0:
            ranked_papers = ranked_papers[:top_n]

        return ranked_papers

    def get_problem_rankings(self, problem_id: int, include_zero_scores: bool = False) -> List[Dict[str, Any]]:
        """
        获取特定产业难题的完整排名（包括详细评分信息）

        Args:
            problem_id: 产业难题ID
            include_zero_scores: 是否包含得分为0的论文

        Returns:
            排序后的论文列表，包含详细评分信息
        """
        if problem_id >= len(self.industry_problems):
            raise ValueError(f"无效的problem_id: {problem_id}")

        rankings = []

        # 遍历所有论文
        for paper_id, paper in self.papers.items():
            # 检查是否有评分记录
            if paper_id in self.scores and problem_id in self.scores[paper_id]:
                score_data = self.scores[paper_id][problem_id]
                score = score_data.get("total_score", 0.0)

                # 构建详细排名信息
                ranking_info = {
                    "paper_id": paper_id,
                    "title": paper.title,
                    "domain": paper.domain,
                    "score": score,
                    "p_score": score_data.get("p_score", 0),
                    "TRL": score_data.get("TRL", 0),
                    "s_score": score_data.get("s_score", 0.0),
                    "result_paper_value": score_data.get("result_paper_value", 0),
                    "result_baseline_value": score_data.get("result_baseline_value", 0),
                    "reasoning": score_data.get("reasoning", ""),
                    "matched": True
                }
                rankings.append(ranking_info)
            elif include_zero_scores:
                # 包含得分为0的论文（匹配但未评分，或未匹配）
                matched = paper_id in self.match_results and self.match_results[paper_id].get(problem_id, False)
                ranking_info = {
                    "paper_id": paper_id,
                    "title": paper.title,
                    "domain": paper.domain,
                    "score": 0.0,
                    "p_score": 0,
                    "TRL": 0,
                    "s_score": 0.0,
                    "result_paper_value": 0,
                    "result_baseline_value": 0,
                    "reasoning": "",
                    "matched": matched
                }
                rankings.append(ranking_info)

        # 按分数从高到低排序
        rankings.sort(key=lambda x: x["score"], reverse=True)

        # 添加排名序号
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1

        return rankings