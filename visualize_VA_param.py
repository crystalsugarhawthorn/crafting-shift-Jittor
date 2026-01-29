import os
import csv
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="选择准则消融可视化（中文，论文风格）")
    parser.add_argument("--dataset", default="PACS", type=str, help="数据集名称")
    parser.add_argument("--backbone", default="resnet18", type=str, help="骨干网络名称")
    parser.add_argument(
        "--exp_names",
        nargs="+",
        default=[
            "imgaug_and_canny_training_all",
            "original_and_canny_training",
            "original-only_training",
        ],
        type=str,
        help="要可视化的实验名称列表",
    )
    parser.add_argument("--results_root", default="Results", type=str, help="结果根目录")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["origin", "worst", "cvar"],
        choices=["origin", "worst", "cvar"],
        help="要对比的选择准则变体",
    )
    parser.add_argument(
        "--focus_outputs",
        nargs="+",
        default=["output_50-50", "output_ImgAug", "output_canny"],
        help="只展示的核心输出头",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("Analysis_Results", "VA_param"),
        type=str,
        help="图片输出目录",
    )
    return parser.parse_args()


def _is_blank_row(row: List[str]) -> bool:
    return not row or all((str(x).strip() == "") for x in row)


def read_total_results_tables(csv_path: str) -> List[Tuple[List[str], List[List[str]]]]:
    tables: List[Tuple[List[str], List[List[str]]]] = []
    if not os.path.isfile(csv_path):
        return tables

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    i = 0
    n = len(rows)
    while i < n:
        while i < n and _is_blank_row(rows[i]):
            i += 1
        if i >= n:
            break

        header = rows[i]
        i += 1
        data_rows: List[List[str]] = []
        while i < n and not _is_blank_row(rows[i]):
            data_rows.append(rows[i])
            i += 1
        tables.append((header, data_rows))

    return tables


def extract_average_test_scores(csv_path: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    tables = read_total_results_tables(csv_path)
    for header, rows in tables:
        if not header or "test_average" not in header or len(header) < 3:
            continue

        output_name = header[2].strip()
        try:
            test_idx = header.index("test_average")
        except ValueError:
            continue

        avg_row = None
        for row in rows:
            if row and row[0].strip().lower() == "average":
                avg_row = row
                break
        if avg_row is None or test_idx >= len(avg_row):
            continue

        try:
            score = float(avg_row[test_idx])
        except ValueError:
            continue
        result[output_name] = score

    return result


def variant_to_dir(results_root: str, variant: str, dataset: str, backbone: str, exp: str) -> str:
    variant_dir = {
        "origin": "Results_origin",
        "worst": "Results_worst",
        "cvar": "Results_cvar",
    }[variant]
    return os.path.join(results_root, variant_dir, f"{dataset}_{backbone}", exp)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _variant_label(variant: str) -> str:
    return {
        "origin": "平均值",
        "worst": "最差组",
        "cvar": "CVaR",
    }[variant]


def _color_for_variant(variant: str) -> str:
    return {
        "origin": "#4C78A8",
        "worst": "#F58518",
        "cvar": "#54A24B",
    }[variant]


def plot_best_bar(best_scores: Dict[str, float], output_dir: str) -> str:
    labels = list(best_scores.keys())
    values = [best_scores[k] for k in labels]
    if not values:
        return ""

    y_min = max(0.0, min(values) - 0.6)
    y_max = max(values) + 0.6

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, values, color=[_color_for_variant(k) for k in labels])

    ax.set_title("最佳测试准确率（不同选择准则）")
    ax.set_ylabel("测试准确率（%）")
    ax.set_xlabel("选择准则")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.22)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.04,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    out_path = os.path.join(output_dir, "图1_最佳准确率.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_focus_grouped(
    variant_scores: Dict[str, Dict[str, float]],
    output_dir: str,
    focus_outputs: List[str],
) -> str:
    variants = list(variant_scores.keys())
    outputs = [o for o in focus_outputs if any(o in s for s in variant_scores.values())]
    if not variants or not outputs:
        return ""

    x = list(range(len(outputs)))
    width = 0.74 / len(variants)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for i, variant in enumerate(variants):
        scores = variant_scores[variant]
        y = [scores.get(o, float("nan")) for o in outputs]
        offset = (i - (len(variants) - 1) / 2) * width
        x_pos = [xi + offset for xi in x]
        ax.bar(
            x_pos,
            y,
            width=width,
            label=_variant_label(variant),
            color=_color_for_variant(variant),
        )

    ax.set_title("核心输出头对比")
    ax.set_ylabel("测试准确率（%）")
    ax.set_xlabel("输出头")
    ax.set_xticks(x)
    ax.set_xticklabels(outputs, rotation=0)
    ax.legend(title=None)
    ax.grid(axis="y", linestyle="--", alpha=0.22)

    out_path = os.path.join(output_dir, "图2_核心输出头对比.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_delta_vs_origin(
    variant_scores: Dict[str, Dict[str, float]],
    output_dir: str,
    focus_outputs: List[str],
) -> str:
    if "origin" not in variant_scores:
        return ""

    origin_scores = variant_scores["origin"]
    variants = [v for v in variant_scores.keys() if v != "origin"]
    outputs = [o for o in focus_outputs if o in origin_scores]
    if not variants or not outputs:
        return ""

    x = list(range(len(outputs)))
    width = 0.68 / len(variants)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for i, variant in enumerate(variants):
        scores = variant_scores[variant]
        delta = [scores.get(o, origin_scores[o]) - origin_scores[o] for o in outputs]
        offset = (i - (len(variants) - 1) / 2) * width
        x_pos = [xi + offset for xi in x]
        ax.bar(
            x_pos,
            delta,
            width=width,
            label=_variant_label(variant),
            color=_color_for_variant(variant),
        )

    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.6)
    ax.set_title("相对提升（相对平均值）")
    ax.set_ylabel("提升（百分点）")
    ax.set_xlabel("输出头")
    ax.set_xticks(x)
    ax.set_xticklabels(outputs, rotation=0)
    ax.legend(title=None)
    ax.grid(axis="y", linestyle="--", alpha=0.22)

    out_path = os.path.join(output_dir, "图3_相对提升.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_best_across_experiments(
    best_by_exp: Dict[str, Dict[str, float]],
    variants: List[str],
    output_dir: str,
) -> str:
    exps = [e for e in best_by_exp.keys() if best_by_exp[e]]
    if not exps:
        return ""

    width = 0.74 / max(len(variants), 1)
    x = list(range(len(exps)))

    all_vals = [best_by_exp[e][v] for e in exps for v in variants if v in best_by_exp[e]]
    if not all_vals:
        return ""
    y_min = max(0.0, min(all_vals) - 0.8)
    y_max = max(all_vals) + 0.8

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    for i, variant in enumerate(variants):
        vals = [best_by_exp[e].get(variant, float("nan")) for e in exps]
        offset = (i - (len(variants) - 1) / 2) * width
        x_pos = [xi + offset for xi in x]
        ax.bar(
            x_pos,
            vals,
            width=width,
            label=_variant_label(variant),
            color=_color_for_variant(variant),
        )

    ax.set_title("不同实验的最佳准确率对比")
    ax.set_ylabel("测试准确率（%）")
    ax.set_xlabel("实验设置")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=8, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.legend(title=None)

    out_path = os.path.join(output_dir, "图4_不同实验对比.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_delta_across_experiments(
    best_by_exp: Dict[str, Dict[str, float]],
    variants: List[str],
    output_dir: str,
) -> str:
    if "origin" not in variants:
        return ""
    exps = [e for e in best_by_exp.keys() if best_by_exp[e]]
    if not exps:
        return ""

    compare_variants = [v for v in variants if v != "origin"]
    if not compare_variants:
        return ""

    width = 0.74 / max(len(compare_variants), 1)
    x = list(range(len(exps)))

    deltas: Dict[str, List[float]] = {}
    all_vals: List[float] = []
    for v in compare_variants:
        cur = []
        for e in exps:
            if "origin" not in best_by_exp[e] or v not in best_by_exp[e]:
                cur.append(float("nan"))
                continue
            d = best_by_exp[e][v] - best_by_exp[e]["origin"]
            cur.append(d)
            all_vals.append(d)
        deltas[v] = cur

    if not all_vals:
        return ""
    y_min = min(all_vals) - 0.2
    y_max = max(all_vals) + 0.2

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    for i, v in enumerate(compare_variants):
        vals = deltas[v]
        offset = (i - (len(compare_variants) - 1) / 2) * width
        x_pos = [xi + offset for xi in x]
        ax.bar(
            x_pos,
            vals,
            width=width,
            label=_variant_label(v),
            color=_color_for_variant(v),
        )

    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.6)
    ax.set_title("不同实验的相对提升（相对平均值）")
    ax.set_ylabel("提升（百分点）")
    ax.set_xlabel("实验设置")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=8, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.legend(title=None)

    out_path = os.path.join(output_dir, "图5_不同实验相对提升.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    setup_chinese_font()
    ensure_dir(args.output_dir)

    print("已生成VA选参可视化：")
    best_by_exp: Dict[str, Dict[str, float]] = {}
    for exp_name in args.exp_names:
        exp_output_dir = os.path.join(args.output_dir, exp_name)
        ensure_dir(exp_output_dir)

        variant_scores: Dict[str, Dict[str, float]] = {}
        best_scores: Dict[str, float] = {}

        for variant in args.variants:
            exp_dir = variant_to_dir(
                results_root=args.results_root,
                variant=variant,
                dataset=args.dataset,
                backbone=args.backbone,
                exp=exp_name,
            )
            total_csv = os.path.join(exp_dir, "Total_results.csv")
            scores = extract_average_test_scores(total_csv)
            if not scores:
                print(f"- [警告] {exp_name}: 未解析到 {total_csv}")
                continue
            variant_scores[variant] = scores

        if not variant_scores:
            print(f"- [跳过] {exp_name}: 没有可用结果。")
            continue

        for variant, scores in variant_scores.items():
            best_scores[variant] = max(scores.values())
        best_by_exp[exp_name] = dict(best_scores)

        p1 = plot_best_bar(best_scores, exp_output_dir)
        p2 = plot_focus_grouped(variant_scores, exp_output_dir, args.focus_outputs)
        p3 = plot_delta_vs_origin(variant_scores, exp_output_dir, args.focus_outputs)
        for p in (p1, p2, p3):
            if p:
                print(f"- {p}")

    p4 = plot_best_across_experiments(best_by_exp, args.variants, args.output_dir)
    p5 = plot_delta_across_experiments(best_by_exp, args.variants, args.output_dir)
    for p in (p4, p5):
        if p:
            print(f"- {p}")


if __name__ == "__main__":
    main()
