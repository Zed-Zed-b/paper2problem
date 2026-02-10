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

        # 缓存难题精简摘要
        self._problems_summary: Optional[List[Dict[str, Any]]] = None

    def _load_prompts(self) -> Dict[str, str]:
        """加载所有prompt文件"""
        prompts_dir = "prompts"
        prompts = {}

        prompt_files = {
            "match_paper": "match_papers.txt",
            "match_patent": "match_patents.txt",
            "match_screening": "match_screening.txt",
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

    async def load_industry_problems(
        self,
        problems_file: str = "new_data/problem/kpi_gen_集成电路_debate_vgemini.json",
        summary_file: str = "new_data/problem/summaries.json"
    ):
        """加载产业难题及其精简摘要"""
        # 异步读取JSON文件
        def read_json_file():
            with open(problems_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        json_data = await asyncio.to_thread(read_json_file)

        problem_list = json_data["results"]
        # processed_problems = []

        # 移除id字段，避免混淆
        for problem in problem_list:
            problem.pop("problem_id", None)
            problem.pop("status", None)
            problem.pop("error_message", None)

        self.industry_problems = problem_list

        # 加载并缓存难题精简摘要（避免重复读取）
        def read_summary_file():
            with open(summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        try:
            self._problems_summary = await asyncio.to_thread(read_summary_file)
        except FileNotFoundError:
            self._problems_summary = None

        # 创建IndustryProblem对象
        for i, problem_data in enumerate(problem_list):
            problem = IndustryProblem(id=str(i), metadata=problem_data)
            self.problems[i] = problem

        return problem_list

    async def load_industry_problems_metric(self, metric_file: str = "集成电路_0125_problems_metric.json"):
        """加载产业难题指标"""
        # 异步读取JSON文件
        def read_json_file():
            with open(metric_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        json_data = await asyncio.to_thread(read_json_file)

        res = json_data["results"]
        metrics = {_item["problem_id"] - 1: _item["metrics"] for _item in res}
        self.industry_problems_metric = metrics
        return metrics

    async def load_paper(self, paper_path: str, paper_type: str = "论文") -> Paper:
        """加载单篇论文并创建Paper对象

        Args:
            paper_path: 论文文件路径
            paper_type: 论文类型，可选"论文"或"专利"，默认为"论文"
        """
        # 异步读取JSON文件
        def read_json_file():
            with open(paper_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        paper_data = await asyncio.to_thread(read_json_file)

        # 从论文数据中提取标题和领域
        if paper_type == "论文":
            title = paper_data.get('title', paper_data.get('paper_title', '未知标题'))
        else:
            title = paper_data.get('title', paper_data["patent_basic_info"].get('patent_title', '未知标题'))
            paper_data.pop("evaluation", None)
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

    async def load_papers_from_directory(self, directory: str, paper_type: str = "论文") -> List[Paper]:
        """从目录加载所有论文

        Args:
            directory: 目录路径
            paper_type: 论文类型，可选"论文"或"专利"，默认为"论文"
        """
        paper_files = glob.glob(os.path.join(directory, "*.json"))
        tasks = [self.load_paper(paper_file, paper_type=paper_type) for paper_file in paper_files]
        papers = await asyncio.gather(*tasks)
        return papers

    async def match_paper_to_problems(self, paper: Paper) -> Dict[int, Any]:
        """
        匹配论文与产业难题（两阶段方案）

        阶段1：初筛 - 使用精简摘要快速筛选候选难题
        阶段2：精筛 - 对候选难题进行详细匹配判断

        Args:
            paper: Paper对象（包含paper_type属性）

        Returns:
            匹配结果字典，problem_id (int) -> {matched (bool), reason (str)}
        """
        paper_type = paper.paper_type

        # 选择精筛阶段的prompt（原有的详细匹配prompt）
        if paper_type == "专利":
            detail_prompt_template = self.prompts.get("match_patent", "")
        else:
            detail_prompt_template = self.prompts.get("match_paper", "")

        if not detail_prompt_template:
            raise ValueError(f"找不到{paper_type}匹配的prompt模板")

        # 获取论文内容
        paper_content = json.dumps(paper.metadata['raw_data'], ensure_ascii=False, indent=2)

        # ========== 阶段1：初筛 ==========
        # 使用缓存的精简摘要
        problems_summary = self._problems_summary

        candidate_ids = list(range(len(self.industry_problems)))  # 默认全部

        if problems_summary is not None:
            # 加载初筛prompt
            screening_prompt = self.prompts.get("match_screening", "")
            if screening_prompt:
                screening_prompt = screening_prompt.replace("{paper_type}", paper_type)
                screening_prompt = screening_prompt.replace("{paper_content}", paper_content)
                screening_prompt = screening_prompt.replace("{problems_summary}", json.dumps(problems_summary, ensure_ascii=False))

                # 调用LLM进行初筛
                resp = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[{"role": "user", "content": screening_prompt}],
                    temperature=0
                )

                # 解析初筛结果
                screening_result = self.safe_json_loads(resp.choices[0].message.content)
                candidate_ids = screening_result.get("matched_ids", list(range(len(self.industry_problems))))
                # print(f"初筛完成，候选难题数量: {len(candidate_ids)}")
            else:
                print("未找到初筛prompt，回退到详细匹配全部难题")

        # ========== 阶段2：精筛 ==========
        # 只对候选难题进行详细匹配
        # 构建候选难题字典
        industry_problems_dict = {i: self.industry_problems[i] for i in candidate_ids if i < len(self.industry_problems)}

        # 构建详细匹配prompt
        prompt = detail_prompt_template
        prompt = prompt.replace("{paper_type}", paper_type)
        prompt += f"\n\n【{paper_type}内容】\n{paper_content}"
        prompt += f"\n\n【产业难题】\n{json.dumps(industry_problems_dict, ensure_ascii=False, indent=2)}"

        # 调用LLM进行精筛
        resp = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # 解析精筛结果
        result = self.safe_json_loads(resp.choices[0].message.content)

        # 转换为problem_id -> bool的字典
        match_results = {}
        for key, value in result.items():
            try:
                problem_id = int(key)
                # 只有在候选列表中的难题才会被标记为matched
                matched = value.get('matched', False)
                match_reason = value.get('reason', "")
                # 如果不在候选列表中，强制设为False
                if problem_id not in candidate_ids:
                    matched = False
                match_results[problem_id] = {"matched": matched, 
                                             "reason": match_reason}

                # 如果匹配，更新Paper和IndustryProblem对象
                if matched:
                    paper.add_problem(str(problem_id))
                    if problem_id in self.problems:
                        self.problems[problem_id].add_paper(paper, score_data=None)

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
            paper: Paper对象（包含paper_type属性）
            problem_id: 产业难题ID

        Returns:
            评分结果字典，包含level和reason
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

        # 获取产业难题信息
        industry_problem = self.industry_problems[problem_id]

        # 构建完整prompt
        prompt = prompt_template
        
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

        # 提取value和reason
        p_score_dict = score_result.get("p_score", {})
        p_score_value = p_score_dict.get("value", 0) 
        p_score_reason = p_score_dict.get("reason", "")

        trl_dict = score_result.get("trl", {})
        trl_value = trl_dict.get("value", 0) 
        trl_reason = trl_dict.get("reason", "")

        # 根据level计算total_score
        total_score = float(p_score_value) * float(trl_value)

        # 构建完整结果
        result = {
            "p_score": {
                "value": p_score_value,
                "reason": p_score_reason
            },
            "trl": {
                "value": trl_value,
                "reason": trl_reason
            },
            "total_score": total_score,
            "problem_id": problem_id
        }

        # 存储结果
        if paper.paper_id not in self.scores:
            self.scores[paper.paper_id] = {}
        self.scores[paper.paper_id][problem_id] = result

        # 更新IndustryProblem中的得分
        if problem_id in self.problems:
            self.problems[problem_id].update_paper_score_data(paper, result)

        return result

    async def evaluate_metrics_for_paper(self, paper: Paper, problem_id: int) -> Dict[str, Any]:
        """
        评估论文在特定产业难题上的指标

        Args:
            paper: Paper对象（包含paper_type属性）
            problem_id: 产业难题ID

        Returns:
            指标评估结果
        """
        paper_type = paper.paper_type
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

    async def process_paper(
        self,
        paper_path: str,
        paper_type: str = "论文",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        处理单篇论文的完整流程

        Args:
            paper_path: 论文文件路径
            paper_type: 论文类型
            use_cache: 是否使用缓存（默认为True）

        Returns:
            处理结果
        """
        # 加载论文
        paper = await self.load_paper(paper_path, paper_type=paper_type)

        # === 缓存检查：避免重复处理 ===
        _from_cache = False
        if use_cache:
            valid_title = paper.title.replace('/', '_')
            cache_file = f"match_res/{valid_title}.json"
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                    if cached.get("paper_title") == paper.title:
                        cached["_from_cache"] = True
                        return cached
                except (json.JSONDecodeError, IOError):
                    pass  # 缓存文件损坏，忽略并继续处理
        else:
            # 不使用缓存时，删除已有缓存确保重新处理
            valid_title = paper.title.replace('/', '_')
            cache_file = f"match_res/{valid_title}.json"
            if os.path.exists(cache_file):
                os.remove(cache_file)
        # =================================

        # 匹配论文与产业难题
        match_results = await self.match_paper_to_problems(paper)
        matched_ids = [pid for pid, res_dict in match_results.items() if res_dict.get("matched", False)]

        if not matched_ids:
            return {
                # "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "matched": False,
                "matched_problems": [],
                "_from_cache": False,
                # "metric_results": {}
            }

        # 并发评分
        score_tasks = [self.score_paper_for_problem(paper, pid) for pid in matched_ids]
        score_results = await asyncio.gather(*score_tasks)

        # # 并发评估指标
        # metric_tasks = []
        # for pid in matched_ids:
        #     if pid in self.industry_problems_metric:
        #         metric_tasks.append(self.evaluate_metrics_for_paper(paper, pid))

        # metric_results = []
        # if metric_tasks:
        #     metric_results = await asyncio.gather(*metric_tasks)

        # 保存指标结果到文件
        # if metric_results:
        #     os.makedirs("metric_match", exist_ok=True)
        #     output_file = f"metric_match/{paper.paper_id}.json"

        #     # 重新组织数据结构
        #     reorganized_results = []
        #     for result in metric_results:
        #         reorganized_results.append({
        #             "problem_id": result["problem_id"] + 1,
        #             "metrics": result["metrics"]
        #         })

        #     with open(output_file, 'w', encoding='utf-8') as f:
        #         json.dump(reorganized_results, f, ensure_ascii=False, indent=4)

        # 构建最终结果
        matched_problems = []
        for score in score_results:
            pid = score.get("problem_id")
            p_score = score.get("p_score", {}).get("value", 0) # 过滤为匹配度评分为 0 的难题
            if pid is not None and p_score > 0:
                score.pop("problem_id", None)  # 从score中移除problem_id字段
                score.pop("total_score", None)  # 如果不需要在最终结果中展示total_score，可以选择移除
                matched_problems.append({
                    "problem_id": pid,
                    "problem_detail": self.problems[pid].title,
                    "matched_reason": match_results.get(pid, {}).get("reason", ""),
                    "score": score,                    
                    }
                )
        
        if not matched_problems:
            return {
                # "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "matched": False,
                "matched_problems": [],
                "_from_cache": False,
            }

        result = {
            # "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "matched": True,
            "matched_problems": matched_problems,
            "_from_cache": False,
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
        paper_titles = []
        for paper_id, paper in self.papers.items():
            scores = self.get_paper_scores(paper_id)
            matrix.append(scores)
            paper_titles.append(paper.title)

        return matrix, paper_titles

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
        """[DEPRECATED] 此方法已弃用，请使用 get_problem_rankings
        对特定产业难题的论文进行排名（从高到低）

        Args:
            problem_id: 产业难题ID
            top_n: 返回前N名，如果为None则返回所有

        Returns:
            排序后的论文列表，每个元素包含paper_id、title、score等信息
        """
        import warnings
        warnings.warn(
            "rank_papers_for_problem is deprecated, use get_problem_rankings instead",
            DeprecationWarning,
            stacklevel=2
        )
        if problem_id >= len(self.industry_problems):
            raise ValueError(f"无效的problem_id: {problem_id}")

        # 检查难题是否存在
        if problem_id not in self.problems:
            return []

        problem = self.problems[problem_id]

        # 获取排序后的论文（包含None得分的论文）
        ranked_pairs = problem.get_ranked_papers(include_none_scores=True)

        ranked_papers = []
        for paper, score_data in ranked_pairs:

            # 构建论文信息
            paper_info = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "domain": paper.domain,
                "score_data": score_data,
                "matched": True  # 因为来自problem.belonged_papers，所以肯定是匹配的
            }

            ranked_papers.append(paper_info)

        # 限制返回数量
        if top_n is not None and top_n > 0:
            ranked_papers = ranked_papers[:top_n]

        return ranked_papers