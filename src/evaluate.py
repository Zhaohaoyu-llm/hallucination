# -*- coding: utf-8 -*-
"""
评估模块：对比检测结果与 ground_truth 人工标注
计算检出率（召回率）、精确率、F1 分数，并分析误判 case
"""

import json
import os
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DetectionResult


@dataclass
class EvalMetrics:
    """评估指标"""
    total: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total > 0 else 0.0


@dataclass
class MismatchCase:
    id: str
    mismatch_type: str  # "false_positive" | "false_negative" | "type_mismatch"
    detail: str


class Evaluator:

    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            self.ground_truth: List[dict] = json.load(f)
        self.gt_map: Dict[str, dict] = {g["id"]: g for g in self.ground_truth}

    def evaluate(self, detection_results: List[DetectionResult]) -> Tuple[EvalMetrics, List[MismatchCase]]:
        metrics = EvalMetrics(total=len(detection_results))
        mismatches: List[MismatchCase] = []

        for dr in detection_results:
            gt = self.gt_map.get(dr.id)
            if gt is None:
                continue

            gt_is_hall = gt["is_hallucination"]
            gt_type = gt.get("hallucination_type")

            if gt_is_hall and dr.is_hallucination:
                metrics.true_positive += 1
                if dr.hallucination_type != gt_type:
                    mismatches.append(MismatchCase(
                        id=dr.id,
                        mismatch_type="type_mismatch",
                        detail=f"Detected='{dr.hallucination_type}', GT='{gt_type}'. Detail: {dr.detail}"
                    ))
            elif not gt_is_hall and not dr.is_hallucination:
                metrics.true_negative += 1
            elif not gt_is_hall and dr.is_hallucination:
                metrics.false_positive += 1
                mismatches.append(MismatchCase(
                    id=dr.id,
                    mismatch_type="false_positive",
                    detail=f"False Positive! GT=normal, Detected='{dr.hallucination_type}'. Detail: {dr.detail}"
                ))
            elif gt_is_hall and not dr.is_hallucination:
                metrics.false_negative += 1
                mismatches.append(MismatchCase(
                    id=dr.id,
                    mismatch_type="false_negative",
                    detail=f"False Negative! GT={gt_type}, Detected=normal. GT Detail: {gt['detail']}"
                ))

        return metrics, mismatches

    def print_report(self, metrics: EvalMetrics, mismatches: List[MismatchCase],
                     results: List[DetectionResult]):
        print("\n" + "=" * 70)
        print("               Hallucination Detection -- Evaluation Report")
        print("=" * 70)

        # 1. Confusion Matrix
        print("\n[CONFUSION MATRIX]")
        print(f"                      Det-Pos     Det-Neg")
        print(f"  Actual-Pos           {metrics.true_positive:>4}         {metrics.false_negative:>4}")
        print(f"  Actual-Neg           {metrics.false_positive:>4}         {metrics.true_negative:>4}")

        # 2. Metrics
        print("\n[METRICS]")
        print(f"  Accuracy:   {metrics.accuracy:.1%}")
        print(f"  Recall:     {metrics.recall:.1%}  (hallucination detection rate)")
        print(f"  Precision:  {metrics.precision:.1%}  (correctness among positives)")
        print(f"  F1 Score:   {metrics.f1:.1%}")

        # 3. Per-item comparison
        print("\n[PER-ITEM COMPARISON]")
        header = f"  {'ID':<6} {'GT':<10} {'Detected':<10} {'Type(Det/GT)':<28} {'Conf':>5}  Match"
        print(header)
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*28} {'-'*5}  {'-'*5}")
        for dr in results:
            gt = self.gt_map.get(dr.id, {})
            gt_hall = "[HALL]" if gt.get("is_hallucination") else "[OK]  "
            dr_hall = "[HALL]" if dr.is_hallucination else "[OK]  "
            type_str = f"{dr.hallucination_type or '-'} / {gt.get('hallucination_type') or '-'}"
            match = "OK" if gt.get("is_hallucination") == dr.is_hallucination else "MISMATCH"
            print(f"  {dr.id:<6} {gt_hall:<10} {dr_hall:<10} {type_str:<28} "
                  f"{dr.confidence:>.0%}   {match}")

        # 4. Mismatch analysis
        if mismatches:
            print(f"\n[MISMATCH ANALYSIS] ({len(mismatches)} cases):")
            fp_cases = [m for m in mismatches if m.mismatch_type == "false_positive"]
            fn_cases = [m for m in mismatches if m.mismatch_type == "false_negative"]
            tm_cases = [m for m in mismatches if m.mismatch_type == "type_mismatch"]

            if fn_cases:
                print(f"\n  [FN] False Negatives ({len(fn_cases)}) -- Missed hallucinations:")
                for c in fn_cases:
                    print(f"     [{c.id}] {c.detail}")

            if fp_cases:
                print(f"\n  [FP] False Positives ({len(fp_cases)}) -- Wrongly flagged:")
                for c in fp_cases:
                    print(f"     [{c.id}] {c.detail}")

            if tm_cases:
                print(f"\n  [TM] Type Mismatch ({len(tm_cases)}) -- Wrong type:")
                for c in tm_cases:
                    print(f"     [{c.id}] {c.detail}")
        else:
            print("\n  [OK] All matches correct, no mismatches!")

        # 5. Per-type recall
        print("\n[PER-TYPE RECALL]")
        type_stats: Dict[str, Dict[str, int]] = {}
        for r in results:
            gt = self.gt_map.get(r.id, {})
            gt_type = gt.get("hallucination_type") or "normal"
            if gt_type not in type_stats:
                type_stats[gt_type] = {"total": 0, "detected": 0}
            type_stats[gt_type]["total"] += 1
            if dr.is_hallucination and gt.get("is_hallucination"):
                type_stats[gt_type]["detected"] += 1

        for tname, stats in sorted(type_stats.items()):
            rate = stats["detected"] / stats["total"] if stats["total"] > 0 else 0
            bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
            print(f"  {tname:<12} {stats['detected']}/{stats['total']:>2}  [{bar}] {rate:.0%}")

        print("\n" + "=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection results")
    parser.add_argument("--results", "-r", default="output/results.json")
    parser.add_argument("--ground-truth", "-g", default="data/ground_truth.json")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        raw = json.load(f)
    results = [DetectionResult(**r) for r in raw]

    evaluator = Evaluator(args.ground_truth)
    metrics, mismatches = evaluator.evaluate(results)
    evaluator.print_report(metrics, mismatches, results)


if __name__ == "__main__":
    main()
