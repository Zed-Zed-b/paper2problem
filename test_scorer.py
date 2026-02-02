 #!/usr/bin/env python3                                                                                                                       
"""                                                                                                                                          
测试Scorer类，输出评分矩阵                                                                                                                   
"""                                                                                                                                          
                                                                                                                                            
import asyncio                                                                                                                               
import sys                                                                                                                                   
from scorer import Scorer                                                                                                                    
                                                                                                                                            
async def main(scorer: Scorer):                                                                                                                            
    """主测试函数"""                                                                                                                                                                                                                                                
                                                                                                                                            
    print("加载产业难题...")                                                                                                                 
    try:                                                                                                                                     
        problems = scorer.load_industry_problems("集成电路_0125_problems.json")                                                              
        print(f"  成功加载 {len(problems)} 个产业难题")                                                                                      
    except FileNotFoundError:                                                                                                                
        print("  错误：找不到产业难题文件")                                                                                                  
        return                                                                                                                               
                                                                                                                                            
    print("加载产业难题指标...")                                                                                                             
    try:                                                                                                                                     
        metrics = scorer.load_industry_problems_metric("集成电路_0125_problems_metric.json")                                                 
        print(f"  成功加载 {len(metrics)} 个难题的指标")                                                                                     
    except FileNotFoundError:                                                                                                                
        print("  警告：找不到产业难题指标文件")                                                                                              
                                                                                                                                            
    # 选择少量论文进行测试（避免过多API调用）                                                                                                
    import glob                                                                                                                              
    paper_dir = "example/英文文献"                                                                                                           
    paper_files = glob.glob(f"{paper_dir}/*.json")[:2]  # 只取前2篇                                                                          
                                                                                                                                            
    if not paper_files:                                                                                                                      
        print(f"错误：在 {paper_dir} 中找不到论文文件")                                                                                      
        return                                                                                                                               
                                                                                                                                            
    print(f"处理 {len(paper_files)} 篇论文...")                                                                                              
                                                                                                                                            
    for i, paper_file in enumerate(paper_files):                                                                                             
        print(f"\n[{i+1}/{len(paper_files)}] 处理论文: {paper_file}")                                                                        
                                                                                                                                            
        try:                                                                                                                                 
            # 处理单篇论文                                                                                                                   
            result = await scorer.process_paper(paper_file, paper_type="论文")                                                               
                                                                                                                                            
            if result["matched"]:                                                                                                            
                print(f"  匹配到 {len(result['matched_problems'])} 个产业难题")                                                              
                print(f"  评分结果: {list(result['scores'].keys())}")                                                                        
            else:                                                                                                                            
                print("  未匹配到任何产业难题")                                                                                              
                                                                                                                                            
        except Exception as e:                                                                                                               
            print(f"  处理失败: {e}")                                                                                                        
            continue                                                                                                                         
                                                                                                                                            
    # 输出评分矩阵                                                                                                                           
    print("\n" + "="*60)                                                                                                                     
    print("评分矩阵:")                                                                                                                       
    print("="*60)                                                                                                                            
                                                                                                                                            
    matrix = scorer.get_all_scores_matrix()                                                                                                  
    paper_names = scorer.get_paper_names()                                                                                                   
    problem_names = scorer.get_problem_names()                                                                                               
                                                                                                                                            
    if not matrix:                                                                                                                           
        print("无评分数据")                                                                                                                  
        return                                                                                                                               
                                                                                                                                            
    print(f"矩阵形状: {len(matrix)} 篇论文 × {len(matrix[0])} 个产业难题")                                                                   
    print()                                                                                                                                  
                                                                                                                                            
    # 打印简化的矩阵（只显示非零值）                                                                                                         
    print("简化矩阵（行：论文，列：产业难题，只显示非零值）:")                                                                               
    for i, scores in enumerate(matrix):                                                                                                      
        non_zero = [(j, score) for j, score in enumerate(scores) if score != 0]                                                              
        if non_zero:                                                                                                                         
            paper_name = paper_names[i] if i < len(paper_names) else f"论文{i}"                                                              
            print(f"  {paper_name[:50]}...")                                                                                                 
            for problem_id, score in non_zero:                                                                                               
                problem_name = problem_names[problem_id] if problem_id < len(problem_names) else f"问题{problem_id}"                         
                print(f"    问题{problem_id}: {problem_name[:40]}... = {score:.4f}")                                                         
                                                                                                                                            
    # 打印完整的数值矩阵                                                                                                                     
    # print("\n完整数值矩阵:")                                                                                                                 
    # print("[" + ",\n ".join([str(row) for row in matrix]) + "]")                                                                             
                                                                                                                                            
    # # 保存矩阵到文件                                                                                                                         
    # output_file = "score_matrix.json"                                                                                                        
    # import json                                                                                                                              
    # with open(output_file, 'w', encoding='utf-8') as f:                                                                                      
    #     json.dump({                                                                                                                          
    #         "paper_names": paper_names,                                                                                                      
    #         "problem_names": problem_names,                                                                                                  
    #         "matrix": matrix                                                                                                                 
    #     }, f, ensure_ascii=False, indent=2)                                                                                                  
    # print(f"\n评分矩阵已保存到: {output_file}")                                                                                              
                                                                                                                                               
if __name__ == "__main__":  
    print("初始化Scorer...")                                                                                                                 
    scorer = Scorer()                                                                                                                  
    asyncio.run(main(scorer=scorer))

    pass