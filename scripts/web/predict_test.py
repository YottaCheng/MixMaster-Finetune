#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
import time
from colorama import init, Fore, Style

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# 直接从当前目录导入
from predict import MixingLabelPredictor

def print_colored(text, color=Fore.WHITE, end='\n'):
    """打印彩色文本"""
    print(f"{color}{text}{Style.RESET_ALL}", end=end)

def run_boundary_tests(predictor, lang, temperature):
    """运行边界条件测试"""
    print_colored("\n----- 边界条件测试 -----", Fore.MAGENTA)
    
    # 边界测试用例
    test_cases = [
        {"name": "空输入测试", "input": ""},
        {"name": "单词测试", "input": "清晰"},
        {"name": "单字母测试", "input": "s"},
        {"name": "极短输入测试", "input": "增加低频"},
        {"name": "极长输入测试", "input": "我想处理一段人声录音，这段录音是在我家的卧室录制的，使用了一个电容麦克风，录音环境不是很理想，有一些房间混响。声音听起来有点闷，高频不够亮，同时低频有点模糊，感觉缺乏存在感。我希望这段人声能够在混音中更加突出，更有穿透力，但又不要有刺耳感。我用的是Logic Pro X，不知道应该如何调节均衡器和压缩器参数。另外，我也想知道是否需要添加一些混响或者延迟效果来增加空间感？"}
    ]
    
    # 执行测试
    for test in test_cases:
        print_colored(f"\n执行: {test['name']}", Fore.YELLOW)
        print_colored(f"输入: '{test['input']}'", Fore.CYAN)
        
        try:
            # 如果是空输入，显示提示并跳过
            if not test['input'].strip():
                print_colored("检测到空输入，模型将无法处理", Fore.YELLOW)
                continue
                
            # 测量处理时间
            start_time = time.time()
            main_label, secondary_label, code = predictor.predict(test['input'], lang, temperature)
            duration = time.time() - start_time
            
            # 输出结果
            print_colored(f"处理时间: {duration:.2f}秒", Fore.BLUE)
            print_colored(f"主标签: {main_label}", Fore.GREEN)
            if secondary_label:
                print_colored(f"副标签: {secondary_label}", Fore.GREEN)
            print_colored(f"标签代码: {code}", Fore.BLUE)
            
        except Exception as e:
            print_colored(f"测试错误: {str(e)}", Fore.RED)
    
    print_colored("\n边界条件测试完成", Fore.MAGENTA)

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
    parser.add_argument('--test-boundary', action='store_true', 
                        help='运行边界条件测试')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print_colored("\n🎚️ 混音效果智能分类系统 - 终端版本", Fore.CYAN)
    print_colored("-------------------------------", Fore.CYAN)
    
    try:
        # 初始化预测器
        print_colored("正在加载模型...", Fore.YELLOW)
        predictor = MixingLabelPredictor(model_dir=args.model)
        print_colored("✅ 模型加载成功！\n", Fore.GREEN)
        
        # 如果指定了边界测试，则执行测试后退出
        if args.test_boundary:
            run_boundary_tests(predictor, args.lang, args.temperature)
            return 0
        
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
                print_colored("检测到空输入，请重新输入", Fore.YELLOW)
                continue
            
            # 执行预测
            print_colored("正在分析...", Fore.YELLOW)
            start_time = time.time()
            main_label, secondary_label, code = predictor.predict(user_input, args.lang, args.temperature)
            duration = time.time() - start_time
            
            # 输出结果
            print_colored("\n----- 分析结果 -----", Fore.CYAN)
            print_colored(f"处理时间: {duration:.2f}秒", Fore.BLUE)
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