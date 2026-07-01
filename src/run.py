# -*- coding: utf-8 -*-
"""
主入口：运行幻觉检测 + 评估
用法: python src/run.py [--mode mock|real] [--api-key KEY]
"""

import json
import os
import sys
import argparse
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import create_detector
from evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description="客服回复幻觉检测工具")
    parser.add_argument("--mode", "-m", default="mock", choices=["mock", "real"],
                        help="检测模式: mock(规则引擎) 或 real(Anthropic API)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API Key (real 模式必需)")
    parser.add_argument("--replies", "-r", default="data/replies.json",
                        help="待检测回复文件路径")
    parser.add_argument("--ground-truth", "-g", default="data/ground_truth.json",
                        help="人工标注 ground truth 文件路径")
    parser.add_argument("--output", "-o", default="output/results.json",
                        help="检测结果输出路径")
    parser.add_argument("--skip-eval", action="store_true",
                        help="跳过评估，仅输出检测结果")
    args = parser.parse_args()

    # 加载数据
    print(f"[LOAD] 加载数据: {args.replies}")
    with open(args.replies, "r", encoding="utf-8") as f:
        replies = json.load(f)
    print(f"       共 {len(replies)} 条客服回复待检测")

    # 创建检测器
    print(f"\n[MODE] 检测模式: {args.mode.upper()}")
    mode_desc = {
        "mock": "规则引擎（基于关键字和模式匹配的启发式检测）",
        "real": "Anthropic Claude API（基于大语言模型的语义理解检测）",
    }
    print(f"       {mode_desc.get(args.mode, '')}")

    detector = create_detector(mode=args.mode, api_key=args.api_key)

    # 执行检测
    print(f"\n[RUN] 开始检测...")
    results = detector.detect_all(replies)

    # 保存结果
    print(f"\n[SAVE] 保存结果: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # 统计概览
    hallucination_count = sum(1 for r in results if r.is_hallucination)
    print(f"\n[STATS] 检测概览:")
    print(f"        总计: {len(results)} 条")
    print(f"        检出幻觉: {hallucination_count} 条 ({hallucination_count / len(results):.0%})")
    print(f"        判定正常: {len(results) - hallucination_count} 条")

    # 评估
    if not args.skip_eval:
        print(f"\n[EVAL] 对比 ground_truth 进行评估...")
        evaluator = Evaluator(args.ground_truth)
        metrics, mismatches = evaluator.evaluate(results)
        evaluator.print_report(metrics, mismatches, results)


if __name__ == "__main__":
    main()
