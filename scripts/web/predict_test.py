#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
from colorama import init, Fore, Style

# 导入预测器类
from scripts.web.predict import MixingLabelPredictor

def print_colored(text, color=Fore.WHITE, end='\n'):
    """打印彩色文本"""
    print(f"{color}{text}{Style.RESET_ALL}", end=end)

def main():
    """主函数"""
    # 初始化colorama
    init()
    
    # 设置参数解析
    parser = argparse.ArgumentParser(description='音频处理需求标签预测 - 终端版本')
    parser.add_argument('-m', '--model', default=r"D:\kings\prj\Finetune_local\Models\deepseek_R1_MixMaster\v6", 
                        help='模型目录路径')
    parser.add_argument('-i', '--input', help='直接输入处理需求文本')
    parser.add_argument('-l', '--lang', choices=['中文', 'English'], default='中文',
                        help='输出语言 (默认: 中文)')
    parser.add_argument('-t', '--temperature', type=float, default=0.1,
                        help='生成温度 (默认: 0.1)')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print_colored("\n🎚️ 混音效果智能分类系统 - 终端版本", Fore.CYAN)
    print_colored("-------------------------------", Fore.CYAN)
    
    try:
        # 初始化预测器
        print_colored("正在加载模型...", Fore.YELLOW)
        predictor = MixingLabelPredictor(model_dir=args.model)
        print_colored("✅ 模型加载成功！\n", Fore.GREEN)
        
        # 交互模式
        while True:
            # 获取输入
            if args.input:
                user_input = args.input
                print_colored(f"输入: {user_input}", Fore.CYAN)
                args.input = None  # 只在第一次循环使用命令行输入
            else:
                print_colored("请输入音频处理需求 (输入'退出'或'exit'结束程序):", Fore.CYAN)
                user_input = input("> ")
            
            # 检查退出指令
            if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                print_colored("\n感谢使用，再见！", Fore.CYAN)
                break
                
            # 跳过空输入
            if not user_input.strip():
                continue
            
            # 执行预测
            print_colored("正在分析...", Fore.YELLOW)
            main_label, secondary_label, code = predictor.predict(user_input, args.lang)
            
            # 输出结果
            print_colored("\n----- 分析结果 -----", Fore.CYAN)
            print_colored(f"主标签: {main_label}", Fore.GREEN)
            if secondary_label:
                print_colored(f"副标签: {secondary_label}", Fore.GREEN)
            print_colored(f"标签代码: {code}", Fore.BLUE)
            print_colored("-------------------\n", Fore.CYAN)
            
    except Exception as e:
        print_colored(f"❌ 错误: {str(e)}", Fore.RED)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())