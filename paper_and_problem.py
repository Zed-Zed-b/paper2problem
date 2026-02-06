from typing import List, Tuple, Optional, Dict, Any


class Paper:
    def __init__(self, metadata, paper_type: str = "论文"):
        """
        论文/专利类

        Args:
            metadata: 包含论文元数据的字典，应包含'title'和'domain'字段
            paper_type: 论文类型，可选"论文"或"专利"，默认为"论文"
        """
        self.title = metadata.get('title', '未知标题')
        self.domain = metadata.get('domain', '未知领域')
        self.paper_type = paper_type
        self.metadata = metadata
        self.belonged_problems = []  # 匹配的产业难题ID列表
        self.paper_id = None  # 论文ID，通常为文件名
        self.file_path = None  # 文件路径

    def __repr__(self):
        return f"Paper(title='{self.title}', domain='{self.domain}', type='{self.paper_type}')"

    def __str__(self):
        return self.title

    def add_problem(self, problem_id: str):
        """添加匹配的产业难题"""
        if problem_id not in self.belonged_problems:
            self.belonged_problems.append(problem_id)

    def get_problem_ids(self) -> List[str]:
        """获取匹配的产业难题ID列表"""
        return self.belonged_problems.copy()

    def has_problem(self, problem_id: str) -> bool:
        """检查是否匹配指定的产业难题"""
        return problem_id in self.belonged_problems


class IndustryProblem:
    def __init__(self, id: str, metadata):
        """
        产业难题类

        Args:
            id: 难题ID（字符串）
            metadata: 难题元数据字典
        """
        self.id = id
        self.metadata = metadata
        self.belonged_papers: List[Tuple['Paper', Optional[Dict[str, Any]]]] = []  # 匹配的论文及评分数据列表，评分数据可能为None
        self.title = metadata.get('title', f'问题{id}')

    def __repr__(self):
        return f"IndustryProblem(id='{self.id}', title='{self.title}')"

    def __str__(self):
        return self.title

    def add_paper(self, paper: Paper, score_data: Optional[Dict[str, Any]] = None):
        """添加匹配的论文及评分数据

        Args:
            paper: Paper对象
            score_data: 完整的评分数据字典，包含total_score等字段，如果为None表示未评分
        """
        # 检查是否已存在
        for i, (existing_paper, existing_data) in enumerate(self.belonged_papers):
            if existing_paper == paper:
                # 更新评分数据
                self.belonged_papers[i] = (paper, score_data)
                return

        # 不存在则添加
        self.belonged_papers.append((paper, score_data))
        paper.add_problem(self.id)

    def get_papers(self) -> List['Paper']:
        """获取匹配的论文列表（不包含得分）"""
        return [paper for paper, _ in self.belonged_papers]

    def get_papers_with_scores(self) -> List[Tuple['Paper', Optional[Dict[str, Any]]]]:
        """获取匹配的论文及评分数据列表"""
        return self.belonged_papers.copy()

    def has_paper(self, paper: Paper) -> bool:
        """检查是否匹配指定的论文"""
        return any(p == paper for p, _ in self.belonged_papers)

    def get_paper_count(self) -> int:
        """获取匹配的论文数量"""
        return len(self.belonged_papers)

    def get_ranked_papers(self, include_none_scores: bool = False) -> List[Tuple['Paper', Optional[Dict[str, Any]]]]:
        """[DEPRECATED] 此方法已弃用，请使用基于level的排名逻辑
        获取按得分从高到低排序的论文列表

        Args:
            include_none_scores: 是否包含评分数据为None的论文，如果为False则只包含有评分数据的论文

        Returns:
            排序后的(论文, 评分数据)列表，按total_score从高到低排序，None评分数据排在最后（如果包含）
        """
        import warnings
        warnings.warn(
            "get_ranked_papers is deprecated, ranking is now based on level",
            DeprecationWarning,
            stacklevel=2
        )
        if include_none_scores:
            papers = self.belonged_papers.copy()
        else:
            papers = [(paper, score_data) for paper, score_data in self.belonged_papers if score_data is not None]

        # 排序：total_score高的在前，None评分数据排在最后
        papers.sort(key=lambda x: (
            x[1] is None,  # None数据排在最后
            -x[1].get('total_score', 0.0) if x[1] is not None else 0  # 按total_score降序
        ))
        return papers

    def get_paper_level(self, paper: Paper) -> Optional[int]:
        """获取指定论文的等级（从评分数据中提取level）

        Returns:
            level值 (1-4)，L4=4分最高，L1=1分最低，如果无评分数据则返回None
        """
        for p, score_data in self.belonged_papers:
            if p == paper:
                if score_data is not None:
                    return score_data.get('level', 0)
                else:
                    return None
        return None

    def update_paper_score_data(self, paper: Paper, score_data: Optional[Dict[str, Any]]) -> bool:
        """更新指定论文的评分数据，如果论文不存在则添加"""
        for i, (p, _) in enumerate(self.belonged_papers):
            if p == paper:
                self.belonged_papers[i] = (paper, score_data)
                return True
        # 论文不存在，添加它
        self.add_paper(paper, score_data)
        return True

    def get_paper_score_data(self, paper: Paper) -> Optional[Dict[str, Any]]:
        """获取指定论文的完整评分数据"""
        for p, score_data in self.belonged_papers:
            if p == paper:
                return score_data
        return None