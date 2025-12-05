import sys
import os
from bs4 import BeautifulSoup

def parse_html_file(filepath):
    """
    解析单个HTML文件，提取Metrics和Examples
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')
    
    # --- 1. 提取 Metrics ---
    metrics = {}
    table = soup.find('table')
    if table:
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                val = cols[1].get_text(strip=True)
                if key.lower() != 'metric': # 跳过表头
                    metrics[key] = val

    # --- 2. 提取 Examples ---
    examples = {}
    
    # 找到 Examples 标题后的所有内容
    examples_start = content.find('<h1>Examples</h1>')
    if examples_start == -1:
        print(f"Warning: No Examples found in {filepath}")
        return metrics, examples
        
    examples_content = content[examples_start:]
    # 使用 <hr> 分割
    blocks = examples_content.split('<hr>')
    
    for block in blocks:
        block_soup = BeautifulSoup(block, 'html.parser')
        
        # 提取 User Prompt
        user_div = block_soup.find('div', class_='message user')
        if not user_div:
            continue
        prompt = user_div.find('pre').get_text(strip=True)
        
        # 提取 Assistant Response
        assist_div = block_soup.find('div', class_='message assistant')
        response = assist_div.find('pre').get_text() if assist_div else "No response"
        
        # 提取 Score 和 Correct Answer
        # 这里的逻辑是查找Results标题下的p标签
        score = "N/A"
        correct_answer = "N/A"
        extracted_answer = "N/A"
        
        # 解析 Results 部分
        # 寻找包含 "Score:" 的文本
        for p in block_soup.find_all('p'):
            text = p.get_text(strip=True)
            if text.startswith("Score:"):
                score = text.replace("Score:", "").strip()
            elif text.startswith("Correct Answer:"):
                correct_answer = text.replace("Correct Answer:", "").strip()
            elif text.startswith("Extracted Answer:"):
                extracted_answer = text.replace("Extracted Answer:", "").strip()

        # 标准化 Score 为布尔值 (根据你的示例 False/True)
        is_correct = False
        if score.lower() == 'true' or score == '1.0' or score == '1':
            is_correct = True
            
        examples[prompt] = {
            'response': response,
            'score_raw': score,
            'is_correct': is_correct,
            'correct_answer': correct_answer,
            'extracted_answer': extracted_answer
        }
        
    return metrics, examples

def generate_comparison_html(file1_path, file2_path, output_path):
    print(f"Parsing {file1_path}...")
    m1, e1 = parse_html_file(file1_path)
    print(f"Parsing {file2_path}...")
    m2, e2 = parse_html_file(file2_path)
    
    # --- 分类比较 ---
    common_prompts = set(e1.keys()) & set(e2.keys())
    
    regressed = [] # 之前对，现在错
    improved = []  # 之前错，现在对
    both_wrong = []
    both_correct = []
    
    for prompt in common_prompts:
        item1 = e1[prompt]
        item2 = e2[prompt]
        
        if item1['is_correct'] and not item2['is_correct']:
            regressed.append(prompt)
        elif not item1['is_correct'] and item2['is_correct']:
            improved.append(prompt)
        elif not item1['is_correct'] and not item2['is_correct']:
            both_wrong.append(prompt)
        else:
            both_correct.append(prompt)

    # --- 生成 HTML ---
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Training Diff Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
            h1, h2 {{ color: #333; }}
            .card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            
            /* Metrics Table */
            table.metrics {{ border-collapse: collapse; width: 100%; }}
            table.metrics th, table.metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            table.metrics th {{ background-color: #f2f2f2; }}
            .diff-pos {{ color: green; font-weight: bold; }}
            .diff-neg {{ color: red; font-weight: bold; }}
            
            /* Example Blocks */
            .example-block {{ border: 1px solid #e0e0e0; margin-bottom: 30px; border-radius: 8px; overflow: hidden; }}
            .example-header {{ padding: 10px 15px; font-weight: bold; border-bottom: 1px solid #e0e0e0; }}
            .regressed .example-header {{ background-color: #ffebee; color: #c62828; border-left: 5px solid #c62828; }}
            .improved .example-header {{ background-color: #e8f5e9; color: #2e7d32; border-left: 5px solid #2e7d32; }}
            .neutral .example-header {{ background-color: #eceff1; color: #455a64; border-left: 5px solid #455a64; }}
            
            .prompt-box {{ padding: 15px; background: #fafafa; border-bottom: 1px solid #eee; font-style: italic; color: #555; }}
            
            .comparison-view {{ display: flex; flex-direction: row; }}
            .ver-col {{ flex: 1; padding: 15px; min-width: 0; }} /* min-width 0 fixes pre overflow */
            .ver-col:first-child {{ border-right: 1px solid #eee; }}
            
            .col-header {{ font-weight: bold; margin-bottom: 10px; display: block; color: #777; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #eee; }}
            
            .tag {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px; }}
            .tag-true {{ background: #c8e6c9; color: #256029; }}
            .tag-false {{ background: #ffcdd2; color: #c63737; }}
            
            details {{ margin-top: 10px; cursor: pointer; }}
            summary {{ outline: none; }}
        </style>
    </head>
    <body>
        <h1>Training Comparison Report</h1>
        
        <div class="card">
            <h2>📊 Metrics Comparison</h2>
            <table class="metrics">
                <tr>
                    <th>Metric</th>
                    <th>Before (v1)</th>
                    <th>After (v2)</th>
                    <th>Delta</th>
                </tr>
    """
    
    # 填充 Metrics
    all_metrics = set(m1.keys()) | set(m2.keys())
    for k in sorted(all_metrics):
        v1 = m1.get(k, "N/A")
        v2 = m2.get(k, "N/A")
        
        # 尝试计算数值差异
        delta_str = ""
        try:
            f1 = float(v1)
            f2 = float(v2)
            delta = f2 - f1
            if delta > 0:
                delta_str = f'<span class="diff-pos">+{delta:.4f}</span>'
            elif delta < 0:
                delta_str = f'<span class="diff-neg">{delta:.4f}</span>'
            else:
                delta_str = "0"
        except:
            delta_str = "-"
            
        html += f"<tr><td>{k}</td><td>{v1}</td><td>{v2}</td><td>{delta_str}</td></tr>"
    
    html += """
            </table>
        </div>
        
        <div class="card">
            <h2>📈 Summary of Changes</h2>
            <p>
                <strong style="color: #c62828">Regressed (Correct -> Incorrect):</strong> {len_reg} &nbsp;|&nbsp; 
                <strong style="color: #2e7d32">Improved (Incorrect -> Correct):</strong> {len_imp} &nbsp;|&nbsp;
                <strong>Still Incorrect:</strong> {len_wrong} &nbsp;|&nbsp;
                <strong>Still Correct:</strong> {len_corr}
            </p>
        </div>
    """.format(len_reg=len(regressed), len_imp=len(improved), len_wrong=len(both_wrong), len_corr=len(both_correct))

    # 定义生成单个对比块的函数
    def create_diff_block(prompt, type_class, title_prefix):
        d1 = e1[prompt]
        d2 = e2[prompt]
        
        score1_cls = "tag-true" if d1['is_correct'] else "tag-false"
        score2_cls = "tag-true" if d2['is_correct'] else "tag-false"
        
        return f"""
        <div class="example-block {type_class}">
            <div class="example-header">{title_prefix}</div>
            <div class="prompt-box"><strong>User Prompt:</strong> {prompt}</div>
            <div class="prompt-box" style="font-size:0.9em; background:#fff;">
                <strong>Correct Answer:</strong> {d1['correct_answer']}
            </div>
            <div class="comparison-view">
                <div class="ver-col">
                    <span class="col-header">Version 1 (Before) <span class="tag {score1_cls}">{d1['score_raw']}</span></span>
                    <pre>{d1['response']}</pre>
                    <div style="font-size:0.85em; color:#666; margin-top:5px;">Extracted: {d1['extracted_answer']}</div>
                </div>
                <div class="ver-col">
                    <span class="col-header">Version 2 (After) <span class="tag {score2_cls}">{d2['score_raw']}</span></span>
                    <pre>{d2['response']}</pre>
                    <div style="font-size:0.85em; color:#666; margin-top:5px;">Extracted: {d2['extracted_answer']}</div>
                </div>
            </div>
        </div>
        """

    # --- 重点关注：退步 (Regressed) ---
    if regressed:
        html += '<h2 style="color: #c62828">⚠️ Regressions (Fixed -> Broken)</h2>'
        for p in regressed:
            html += create_diff_block(p, "regressed", "REGRESSION")

    # --- 重点关注：进步 (Improved) ---
    if improved:
        html += '<h2 style="color: #2e7d32">🚀 Improvements (Broken -> Fixed)</h2>'
        for p in improved:
            html += create_diff_block(p, "improved", "IMPROVEMENT")

    # --- 其他类别 (折叠显示) ---
    html += '<details><summary><h2>👀 Persistent Errors (Both Incorrect) - Click to expand</h2></summary>'
    for p in both_wrong:
        html += create_diff_block(p, "neutral", "BOTH INCORRECT")
    html += '</details>'
    
    html += '</body></html>'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Comparison report generated: {output_path}")

if __name__ == "__main__":
    # 使用方法：python script.py file_before.html file_after.html
    if len(sys.argv) < 3:
        print("Usage: python compare.py <before_html> <after_html>")
        print("Example: python compare.py step_100.html step_200.html")
    else:
        generate_comparison_html(sys.argv[1], sys.argv[2], f"output/diff_report_{os.path.basename(sys.argv[1])}_{os.path.basename(sys.argv[2])}.html")