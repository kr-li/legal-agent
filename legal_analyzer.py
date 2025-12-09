# english_contract_analyzer.py
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import gradio as gr
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer,AutoModelForCausalLM
from modelscope import snapshot_download
import numpy as np
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig


class EnglishContractAnalyzer:
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B"):
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model()
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']

    def load_model(self):
        """加载ModelScope模型"""
        print(f"正在加载模型: {self.model_name}")

        try:
            # 下载模型（如果尚未下载）
            model_dir = snapshot_download(self.model_name)

            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )

            print("✅ 模型加载成功！")
            return model, tokenizer

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF提取文本"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"PDF提取失败: {e}")
            # 备用方案：使用pdfminer
            try:
                from pdfminer.high_level import extract_text as pdfminer_extract
                return pdfminer_extract(pdf_path)
            except:
                return f"PDF文本提取失败: {e}"

    def extract_text_from_docx(self, docx_path: str) -> str:
        """从Word文档提取文本"""
        try:
            import docx
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"DOCX提取失败: {e}")
            return f"Word文档提取失败: {e}"

    def extract_text_from_file(self, file_path: str) -> str:
        """根据文件类型提取文本"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"不支持的文件格式: {file_ext}"

    def analyze_contract_clause(self, clause_text: str, analysis_type: str) -> str:
        """分析合同条款"""
        analysis_prompts = {
            "risk_analysis": """Analyze the following contract clause for LEGAL RISKS and provide professional assessment:

CONTRACT CLAUSE:
{clause}

Please analyze in this structure:
1. **RISK IDENTIFICATION** [List specific legal risks]
2. **SEVERITY ASSESSMENT** [High/Medium/Low for each risk]
3. **LEGAL BASIS** [Relevant laws and regulations]
4. **RECOMMENDATIONS** [Specific revision suggestions]
5. **BEST PRACTICES** [Industry standards]

Professional Analysis:""",

            "compliance_check": """Conduct COMPLIANCE REVIEW for the following contract clause:

CLAUSE:
{clause}

Check compliance with:
- General contract law principles
- Industry-specific regulations
- Jurisdictional requirements
- International standards (if applicable)

Provide: Compliance Status + Required Actions + Legal References

Compliance Analysis:""",

            "plain_explanation": """Explain this contract clause in PLAIN ENGLISH for business understanding:

CLAUSE:
{clause}

Please provide:
1. **Simple Explanation** [Clear, non-legal language]
2. **Key Obligations** [What each party must do]
3. **Practical Implications** [Real-world consequences]
4. **Important Considerations** [What to watch out for]

Plain English Explanation:""",

            "full_review": """Comprehensive LEGAL REVIEW of contract clause:

CLAUSE TEXT:
{clause}

Please provide detailed analysis covering:
1. **CLAUSE TYPE & PURPOSE**
2. **KEY TERMS & DEFINITIONS**
3. **LEGAL RISK ASSESSMENT**
4. **COMPLIANCE CHECK**
5. **NEGOTIATION POINTS**
6. **RECOMMENDED REVISIONS**
7. **ALTERNATIVE WORDING**

Comprehensive Legal Review:"""
        }

        prompt_template = analysis_prompts.get(analysis_type, analysis_prompts["risk_analysis"])
        prompt = prompt_template.format(clause=clause_text[:3000])  # 限制长度

        # 生成回答
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Analysis:")[-1].replace(prompt, "").strip()

    def process_uploaded_file(self, file_path: str, analysis_type: str) -> Dict[str, Any]:
        """处理上传的文件"""
        try:
            # 提取文本
            contract_text = self.extract_text_from_file(file_path)

            if "失败" in contract_text or "错误" in contract_text:
                return {
                    "success": False,
                    "error": contract_text,
                    "analysis": ""
                }

            # 分析合同
            analysis = self.analyze_contract_clause(contract_text, analysis_type)

            return {
                "success": True,
                "original_text": contract_text[:1000] + "..." if len(contract_text) > 1000 else contract_text,
                "analysis": analysis,
                "text_length": len(contract_text)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": ""
            }


def create_advanced_interface():
    """创建高级界面"""
    analyzer = EnglishContractAnalyzer()

    with gr.Blocks(theme=gr.themes.Soft(), title="英文合同分析系统") as demo:
        gr.Markdown("# 🇺🇸 英文合同智能分析系统")
        gr.Markdown("基于ModelScope大模型的英文法律合同分析工具")

        with gr.Tabs() as tabs:
            with gr.TabItem("📁 文件分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="上传合同文件",
                            file_types=[".pdf", ".docx", ".doc", ".txt"],
                            type="filepath"
                        )
                        analysis_type = gr.Radio(
                            choices=["risk_analysis", "compliance_check", "plain_explanation", "full_review"],
                            label="分析类型",
                            value="risk_analysis",
                            info="选择分析深度"
                        )
                        analyze_btn = gr.Button("开始分析", variant="primary")

                    with gr.Column(scale=2):
                        original_text = gr.Textbox(
                            label="提取的合同文本",
                            lines=6,
                            max_lines=10,
                            interactive=False
                        )
                        analysis_output = gr.Textbox(
                            label="分析结果",
                            lines=12,
                            interactive=False
                        )
                        file_info = gr.Textbox(
                            label="文件信息",
                            visible=False
                        )

            with gr.TabItem("💬 直接对话"):
                chatbot = gr.Chatbot(label="法律问答对话")
                msg = gr.Textbox(
                    label="输入英文法律问题",
                    placeholder="例如: What are the key risks in this indemnification clause?",
                    lines=3
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

        # 文件分析功能
        def analyze_file(file_path, analysis_type):
            if not file_path:
                return "请先上传文件", "", ""

            result = analyzer.process_uploaded_file(file_path, analysis_type)

            if result["success"]:
                info = f"文本长度: {result['text_length']} 字符"
                return result["original_text"], result["analysis"], info
            else:
                return f"处理失败: {result['error']}", "", ""

        analyze_btn.click(
            analyze_file,
            inputs=[file_input, analysis_type],
            outputs=[original_text, analysis_output, file_info]
        )

        # 对话功能
        def legal_chat(message, chat_history):
            if not message.strip():
                return "", chat_history

            # 构建专业提示词
            prompt = f"""You are a professional legal AI assistant. Please provide accurate, professional analysis for the following legal question.

Question: {message}

Please provide a comprehensive answer with legal basis and practical advice:"""

            inputs = analyzer.tokenizer(prompt, return_tensors="pt").to(analyzer.model.device)
            outputs = analyzer.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.3,
                do_sample=True
            )

            response = analyzer.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("advice:")[-1].strip()

            chat_history.append((message, response))
            return "", chat_history

        send_btn.click(legal_chat, [msg, chatbot], [msg, chatbot])
        msg.submit(legal_chat, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], None, chatbot)

        # 示例部分
        with gr.Accordion("📋 使用示例", open=False):
            gr.Markdown("""
            **示例问题：**
            - What are the key elements of a valid contract?
            - Explain the difference between representation and warranty
            - What risks should I look for in a service agreement?
            - How to negotiate better termination clauses?

            **支持的文件格式：**
            - PDF文档 (.pdf)
            - Word文档 (.docx, .doc)  
            - 文本文件 (.txt)
            """)

    return demo


# 批量处理功能
class BatchContractProcessor:
    """批量合同处理器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def process_batch(self, input_dir: str, output_dir: str, analysis_type: str = "risk_analysis"):
        """批量处理合同文件夹"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        results = []
        supported_files = []

        # 收集支持的文件
        for ext in ['.pdf', '.docx', '.doc', '.txt']:
            supported_files.extend(input_path.glob(f"*{ext}"))

        for file_path in supported_files:
            print(f"处理文件: {file_path.name}")

            try:
                result = self.analyzer.process_uploaded_file(str(file_path), analysis_type)

                if result["success"]:
                    # 保存结果
                    output_file = output_path / f"{file_path.stem}_analysis.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"File: {file_path.name}\n")
                        f.write(f"Analysis Type: {analysis_type}\n")
                        f.write("=" * 50 + "\n")
                        f.write(result["analysis"])

                    results.append({
                        "file": file_path.name,
                        "status": "success",
                        "output_file": str(output_file)
                    })
                else:
                    results.append({
                        "file": file_path.name,
                        "status": "failed",
                        "error": result["error"]
                    })

            except Exception as e:
                results.append({
                    "file": file_path.name,
                    "status": "error",
                    "error": str(e)
                })

        return results


if __name__ == "__main__":
    # 启动服务
    print("🚀 启动英文合同分析系统...")
    print("📊 支持格式: PDF, Word, TXT")
    print("🌐 访问: http://localhost:7860")

    demo = create_advanced_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )