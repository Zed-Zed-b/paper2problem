import json
from openai import OpenAI
import re
import asyncio

client = OpenAI(
    api_key="sk-703b713308054f34b710539c87788d82",
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"

def load_paper(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def safe_json_loads(llm_text: str):
    """安全解析 LLM 返回的 JSON，处理常见的格式问题"""
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
        # 如果失败，尝试提取第一个完整的 JSON 对象
        # 从第一个 { 开始匹配
        start = llm_text.find("{")
        if start == -1:
            raise

        # 计算括号匹配，找到完整的 JSON 对象
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


def _matching_sync(paper, industry_problems, paper_type="论文"):
    industry_problems_dict = {i: prob for i, prob in enumerate(industry_problems)}
    prompt = f"""
你是一位专业的工业界专家以及学者，擅长评估学术研究对产业难题的实际价值。

【任务】
给定一篇{paper_type}，判断该{paper_type}是否对下列产业难题具有实际贡献（即：{paper_type}的研究内容、方法或结果能够为解决该产业难题提供有意义的技术路径、方案或突破）。

【判断维度】
请从以下 3 个维度综合评估{paper_type}是否对产业难题有实际贡献：

1. 研究领域相关性：{paper_type}的研究对象是否与产业难题的核心技术领域高度匹配
2. 应用场景一致性：{paper_type}的应用场景是否与产业难题的工业应用场景一致或相近
3. 问题层面适配度：{paper_type}提出的核心方法所解决的问题是否和产业难题的具体描述匹配或能够推动产业难题的解决

【注意事项】
- 严格匹配：不要因为"都是芯片领域"就认为所有芯片{paper_type}都匹配所有芯片难题
- 仅判断关联性：当前任务只判断{paper_type}与难题是否有匹配，至于解决深度由后续评分步骤评估
- 避免过度泛化：光通信芯片{paper_type}不匹配光刻胶难题，存储芯片{paper_type}不匹配 EDA 工具难题

【{paper_type}内容】
{paper}

【产业难题】
{industry_problems_dict}

【输出要求】
返回一个 **JSON 对象**，键为产业难题的编号，值为一个对象，包含 matched（是否匹配）和 reason（相关理由）;如果匹配，请在 "reason" 字段中输出相关的理由，如果不匹配，请保持 "reason" 字段为空字符串。
请确保你的输出严格遵守 json 格式，可以被正确解析。

例如:
{{"0": {{
    "matched": true,
    "reason": ""
  }},
  "1": {{
    "matched": false,
    "reason": ""
  }},
  "2": {{
    "matched": true,
    "reason": ""
  }}
}}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return safe_json_loads(resp.choices[0].message.content)


def _extract_scores_sync(paper, industry_problem, paper_type="论文"):
    prompt = f"""
你是一位专业的工业界专家以及学者。
给定以下{paper_type}内容和产业难题，请提取该{paper_type}在解决该产业难题方面的能力提升。
这种提升分为三方面：一是性能阈值跨越(p_score)，二是技术成熟度等级（TRL），三是自身表現(s_score)。
p_score 是一个数值，表示该{paper_type}在解决该产业难题方面的性能提升，数值越大表示提升越大。
| 等级 | 名称 | 定义描述 | 关键特征识别（AI / 专家判据） |
|----|----|----|----|
| P1 | 边缘性改进 | 针对难题的周边环节或非核心参数进行了微调，未涉及核心技术逻辑。 | 权利要求多为外围优化；解决的是局部偶发问题；技术可替代性极强。 |
| P2 | 常规性优化 | 采用了行业内通用的技术路径，对特定痛点进行了常规性的工程优化或适配。 | 属于“已知路径”的延伸；实现了性能的稳步增长；未改变原有技术架构。 |
| P3 | 核心瓶颈突破 | 针对产业难题定义的关键瓶颈提出了实质性的创新方案，性能指标有显著提升。 | 触及独立权利要求的核心逻辑；对比现有技术有明确的增量（如功耗降低 20%）；解决了行业共性痛点。 |
| P4 | 系统性创新 | 提出了完整的、系统性的技术新方案，具备强力的国产替代能力或填补了国内空白。 | 涉及底层架构或关键工艺流程的改变；具备较强的技术壁垒；能够支撑完整产品线的自主可控。 |
| P5 | 颠覆式 / 引领性 | 开辟了全新的技术赛道，绕开既有技术封锁，或成为行业标准的基础。 | 基础性 / 源头性专利；迫使竞争对手改变技术路线；或被纳入国家 / 国际技术标准。 |
TRL 是技术成熟度等级（Technology Readiness Level）的缩写，是衡量技术从概念到实际应用的成熟度的指标。TRL 共分为 9 个等级，具体定义如下：
| 阶段 | 等级 | 名称 | 定义与关键特征 |
|----|----|----|----|
| 一、基础研究阶段（发现原理） | TRL 1 | 基本原理被观察到并被报告 | 科学研究的起点。例如：在实验室观察到某种物理现象或数学规律，并发表了初步论文。 |
|  | TRL 2 | 技术概念或应用方案的形成 | 开始将基本原理转向实际应用。例如：提出了某种解决特定产业问题的算法框架。 |
|  | TRL 3 | 关键功能的概念验证（实验室） | 通过分析和实验研究，证明技术方案在逻辑上可行。例如：在公开学术数据集上跑通了 Demo。 |
| 二、技术开发阶段（样机验证） | TRL 4 | 实验室环境下的组件验证 | 将技术组件（代码 / 硬件模块）整合，在控制环境下验证性能。例如：完成了实验室级的原型系统。 |
|  | TRL 5 | 相关环境下的组件 / 样机验证 | 技术在模拟真实工业环境的条件下进行测试。例如：利用企业提供的脱敏生产数据完成离线测试。 |
|  | TRL 6 | 相关环境下的系统级样机演示 | 形成完整系统原型，并在具有代表性的应用场景中运行。例如：在工业现场进行了半实物仿真。 |
| 三、系统验证阶段（产业落地） | TRL 7 | 真实操作环境下的系统演示 | 在实际运行环境（如工厂生产线、真实电网）中进行验证，是解决“最后一公里”的关键。 |
|  | TRL 8 | 最终系统完成并经过测试定型 | 技术完全成熟，通过工业级可靠性与安全性测试。例如：完成芯片流片并进行批量测试。 |
|  | TRL 9 | 实际系统在真实任务中被证明 | 技术已实现商业化大规模应用，或在重大国家工程中稳定运行。例如：列入国家 / 行业采购目录。 |
s_score 是{paper_type}自身取得的在对应产业难题上的性能数值提升。"""+\
r"""它的计算公式为: s_{score} = \left|\frac{Result_{paper} - Result_{baseline}}{Result_{baseline}}\right"""+\
f"""
如果{paper_type}的性能不及基线，请将 result_paper 和 result_baseline 均置为 0。
如果 s_score 无法从{paper_type}中直接提取，即 result_paper 和 result_baseline 有一者缺失，也请将 result_paper 和 result_baseline 均置为 0。
请基于上述定义，提取该{paper_type}在解决以下产业难题方面的 p_score, TRL, 和 s_score（你只需提供 result_paper 和 result_baseline，我来计算s_score）：
{paper_type}内容:
"""+f"""
{paper}
产业难题:
{industry_problem}
请返回一个 JSON 对象，格式如下:
{{
  "p_score": 1~5 的整数,
  "p_score_reason": "详细说明评分依据",
  "TRL": 1~9 的整数,
  "TRL_reason": "详细说明评分依据",
  "result_paper": "浮点数，论文中取得的性能结果",
  "result_baseline": "浮点数，论文中提及的基线性能结果",
  "s_score_reason": "详细说明 result_paper 和 result_baseline 提取结果"
}}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return safe_json_loads(resp.choices[0].message.content)

def _decide_metric_sync(paper, metric_list, paper_type="论文"):
    prompt = f"""
你是一位专业的工业界专家以及学者。
给定以下{paper_type}内容和与此{paper_type}有直接关联 metric 列表，请判断该{paper_type}中是否提供了其提出的方法在某个 metric 上的性能结果。
请返回一个 JSON 对象，格式如下:
{{
    "metric_name_1": {{
        "provided": true/false,
        "value": "浮点数，论文中该 metric 的性能结果, 如果没提供则置为 0",
        "unit": "该 metric 的单位，例如 ms, %, frames per second 等"
        }},
    "metric_name_2": {{
        "provided": true/false,
        "value": "浮点数，论文中该 metric 的性能结果, 如果没提供则置为 0",
        "unit": "该 metric 的单位，例如 ms, %, frames per second 等"
        }}
}}
返回 JSON 对象中的 metric_name 应与输入的 metric_list 中提供的 name 保持一致。

{paper_type}内容:
{paper}
metric 列表:
{metric_list}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return safe_json_loads(resp.choices[0].message.content)


# ---------- async 并发封装 ----------

async def matching(paper, industry_problems, paper_type="论文"):
    return await asyncio.to_thread(_matching_sync, paper, industry_problems, paper_type)


async def extract_scores(paper, industry_problem, paper_type="论文"):
    return await asyncio.to_thread(_extract_scores_sync, paper, industry_problem, paper_type)

async def decide_metric(paper, metric_list, paper_type="论文"):
    return await asyncio.to_thread(_decide_metric_sync, paper, metric_list, paper_type)