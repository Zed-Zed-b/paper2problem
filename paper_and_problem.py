from typing import List


class Paper:
    def __init__(self, metadata):
        """
        论文/专利类

        Args:
            metadata: 包含论文元数据的字典，应包含'title'和'domain'字段
        """
        self.title = metadata.get('title', '未知标题')
        self.domain = metadata.get('domain', '未知领域')
        self.metadata = metadata
        self.belonged_problems = []  # 匹配的产业难题ID列表
        self.paper_id = None  # 论文ID，通常为文件名
        self.file_path = None  # 文件路径

    def __repr__(self):
        return f"Paper(title='{self.title}', domain='{self.domain}')"

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
        self.belonged_papers = []  # 匹配的Paper对象列表
        self.title = metadata.get('title', f'问题{id}')

    def __repr__(self):
        return f"IndustryProblem(id='{self.id}', title='{self.title}')"

    def __str__(self):
        return self.title

    def add_paper(self, paper: Paper):
        """添加匹配的论文"""
        if paper not in self.belonged_papers:
            self.belonged_papers.append(paper)
            paper.add_problem(self.id)

    def get_papers(self) -> List['Paper']:
        """获取匹配的论文列表"""
        return self.belonged_papers.copy()

    def has_paper(self, paper: Paper) -> bool:
        """检查是否匹配指定的论文"""
        return paper in self.belonged_papers

    def get_paper_count(self) -> int:
        """获取匹配的论文数量"""
        return len(self.belonged_papers)