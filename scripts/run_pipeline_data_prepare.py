# /Volumes/Study/prj/scripts/run_pipeline.py
import subprocess
import sys
from pathlib import Path
import json
import os

class EnhancedPipelineRunner:
    def __init__(self):
        # 动态获取项目根目录
        try:
            self.base_dir = Path(__file__).resolve().parent.parent  # 假设脚本在 scripts/ 目录
            print(f"✅ 项目根目录已自动识别: {self.base_dir}")
        except Exception as e:
            print(f"❌ 无法定位项目根目录: {str(e)}")
            sys.exit(1)

        # 初始化路径配置
        self._init_paths()
        
        # 自动创建必要目录
        self._create_directories()

    def _init_paths(self):
        """初始化所有路径配置"""
        self.path_config = {
            # 输入文件
            "input_doc": self.base_dir / "data" / "raw" / "training_raw_data.docx",
            
            # 步骤输出文件
            "outputs": {
                "top_words": self.base_dir / "data" / "processed" / "top_20_words.json",
                "synonyms": self.base_dir / "config" / "music_synonyms.json",
                "eda_results": self.base_dir / "data" / "processed" / "augmented_results.docx",
                "backtrans_results": self.base_dir / "data" / "processed" / "backtrans_results.docx"
            },
            
            # 处理脚本路径
            "scripts": {
                "generate_wordcloud": self.base_dir / "scripts" / "data_prepare" / "generate_wordcloud.py",
                "renew_json": self.base_dir / "scripts" / "data_prepare" / "renew_json.py",
                "pure_eda": self.base_dir / "scripts" / "data_prepare" / "pure_eda.py",
                "backtracking": self.base_dir / "scripts" / "data_prepare" / "backtracking.py",
                #"clean_data": self.base_dir / "scripts" / "data_prepare" / "clean_data.py"
            }
        }

    def _create_directories(self):
        """自动创建缺失目录"""
        required_dirs = [
            self.base_dir / "data" / "raw",
            self.base_dir / "data" / "processed",
            self.base_dir / "config",
            self.base_dir / "scripts" / "data_prepare"
        ]
        
        for directory in required_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"📂 已创建/确认目录: {directory.relative_to(self.base_dir)}")
            except Exception as e:
                print(f"❌ 目录创建失败: {directory} - {str(e)}")
                sys.exit(1)

    def _validate_inputs(self):
        """验证输入文件存在"""
        missing = []
        print("\n🔍 输入文件验证：")
        
        # 检查原始数据文件
        input_doc = self.path_config["input_doc"]
        status = "✅ 存在" if input_doc.exists() else "❌ 缺失"
        print(f"训练数据文件".ljust(20) + f" | {status} | {input_doc.relative_to(self.base_dir)}")
        
        if not input_doc.exists():
            missing.append(input_doc)
        
        # 检查脚本文件
        print("\n🔧 脚本文件验证：")
        for script_name, script_path in self.path_config["scripts"].items():
            status = "✅ 存在" if script_path.exists() else "❌ 缺失"
            print(f"{script_name.ljust(20)} | {status} | {script_path.relative_to(self.base_dir)}")
            if not script_path.exists():
                missing.append(script_path)
        
        if missing:
            print(f"\n❌ 缺失 {len(missing)} 个关键文件：")
            for path in missing:
                print(f"  - {path.relative_to(self.base_dir)}")
            sys.exit(1)

    def _run_pipeline_step(self, step: dict) -> bool:
        """执行单个流水线步骤"""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1" 
        try:
            print(f"\n🚀 开始执行: {step['name']}")
            print("━" * 50)
            
            # 显示完整命令
            cmd_str = " ".join(str(arg) for arg in step["command"])
            print(f"执行命令: {cmd_str}")
            
            result = subprocess.run(
                step["command"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            print("输出日志:")
            print(result.stdout)
            print("错误日志:")  # 新增：显式打印错误日志
            print(result.stderr)  # 确保 stderr 被捕获
            
            # 验证输出文件
            missing_outputs = [f for f in step["outputs"] if not f.exists()]
            if missing_outputs:
                print(f"⚠️ 未生成预期输出文件:")
                for f in missing_outputs:
                    print(f"  - {f.relative_to(self.base_dir)}")
                return False
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"""
            ❌ 执行失败: {step['name']}
            错误代码: {e.returncode}
            错误详情:
            {e.stderr.strip() or '无额外错误信息'}
            """)
            return False
        except FileNotFoundError:
            print(f"""
            ❌ 找不到脚本文件: {step['command'][1]}
            预期路径: {step['command'][1]}
            实际路径: {Path(step['command'][1]).resolve() if Path(step['command'][1]).exists() else '文件不存在'}
            """)
            return False

    def _get_pipeline_steps(self):
        """定义流水线步骤"""
        return [
            {
                "name": "生成词云数据",
                "command": [
                    sys.executable,
                    str(self.path_config["scripts"]["generate_wordcloud"]),
                    "--input", str(self.path_config["input_doc"]),
                    "--output", str(self.path_config["outputs"]["top_words"])
                ],
                "outputs": [self.path_config["outputs"]["top_words"]],
                "troubleshooting": [
                    "确认输入文件包含有效文本内容",
                    "检查Python环境依赖是否安装"
                ]
            },
            {
                "name": "生成替换词典",
                "command": [
                    sys.executable,
                    str(self.path_config["scripts"]["renew_json"]),
                    "--input", str(self.path_config["outputs"]["top_words"]),
                    "--output", str(self.path_config["outputs"]["synonyms"])
                ],
                "outputs": [self.path_config["outputs"]["synonyms"]],
                "troubleshooting": [
                    "确认词云文件格式正确",
                    "检查JSON写入权限"
                ]
            },
            {
                "name": "执行EDA增强",
                "command": [
                    sys.executable,
                    str(self.path_config["scripts"]["pure_eda"]),
                    "--input", str(self.path_config["input_doc"]),
                    "--config", str(self.path_config["outputs"]["synonyms"]),
                    "--output", str(self.path_config["outputs"]["eda_results"])
                ],
                "outputs": [self.path_config["outputs"]["eda_results"]],
                "troubleshooting": [
                    "验证同义词词典格式",
                    "检查输出目录写入权限"
                ]
            },
            {
                "name": "执行回译增强",
                "command": [
                    sys.executable,
                    str(self.path_config["scripts"]["backtracking"]),
                    "--input", str(self.path_config["input_doc"]),
                    "--output", str(self.path_config["outputs"]["backtrans_results"])
                ],
                "outputs": [self.path_config["outputs"]["backtrans_results"]],
                "troubleshooting": [
                    "确认翻译API密钥有效",
                    "检查网络连接"
                ]
            }
        ]

    def run(self):
        print("\n" + "="*50)
        print("🚀 启动音乐数据处理流水线")
        print("="*50)
        
        # 前置检查
        self._validate_inputs()
        
        # 执行流水线
        total_steps = len(self._get_pipeline_steps())
        for step_num, step_config in enumerate(self._get_pipeline_steps(), 1):
            print(f"\n🔧 正在执行步骤 {step_num}/{total_steps}")
            success = self._run_pipeline_step(step_config)
            
            if not success:
                print("\n" + "!"*50)
                print(f"❌ 流水线在步骤 [{step_config['name']}] 失败")
                print("建议排查步骤:")
                for tip in step_config.get("troubleshooting", []):
                    print(f"  - {tip}")
                sys.exit(1)
        
        # 最终验证
        print("\n" + "="*50)
        print("🔍 最终输出验证")
        all_outputs = [
            self.path_config["outputs"]["top_words"],
            self.path_config["outputs"]["synonyms"],
            self.path_config["outputs"]["eda_results"],
            self.path_config["outputs"]["backtrans_results"]
        ]
        
        missing = [f for f in all_outputs if not f.exists()]
        if missing:
            print(f"⚠️ 缺失 {len(missing)} 个输出文件:")
            for f in missing:
                print(f"  - {f.relative_to(self.base_dir)}")
            sys.exit(1)
        
        print("\n" + "✅"*3 + " 流水线执行成功 " + "✅"*3)
        print(f"生成文件位置: {self.base_dir/'data'/'processed'}")
        print(f"总生成文件数: {len(all_outputs)}")
        print("="*50)

if __name__ == "__main__":
    EnhancedPipelineRunner().run()