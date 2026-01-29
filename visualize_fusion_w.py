import os
import csv
import argparse
from typing import Dict, List, Tuple
from statistics import mean, stdev

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Fusion w 对比可视化（中文，论文风格）")
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
        help="要对比的实验名称列表",
    )
    parser.add_argument("--results_root", default="Results", type=str, help="结果根目录")
    parser.add_argument(
        "--origin_dir",
        default="Results_origin",
        type=str,
        help="优化前结果目录名",
    )
    parser.add_argument(
        "--fusion_dir",
        default="Results_Fusion_w",
        type=str,
        help="优化后结果目录名",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("Analysis_Results", "fusion_w"),
        type=str,
        help="图片输出目录",
    )
    return parser.parse_args()


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


def _is_blank_row(row: List[str]) -> bool:
    return (not row) or all(str(x).strip() == "" for x in row)


def read_total_results_tables(csv_path: str) -> List[Tuple[List[str], List[List[str]]]]:
    tables: List[Tuple[List[str], List[List[str]]]] = []
    if not os.path.isfile(csv_path):
        return tables

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    i, n = 0, len(rows)
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


def total_csv_path(results_root: str, variant_dir: str, dataset: str, backbone: str, exp: str) -> str:
    return os.path.join(results_root, variant_dir, f"{dataset}_{backbone}", exp, "Total_results.csv")


def extract_average_test_scores(csv_path: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for header, rows in read_total_results_tables(csv_path):
        if "test_average" not in header or len(header) < 3:
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
            result[output_name] = float(avg_row[test_idx])
        except ValueError:
            continue
    return result


def extract_seed_scores_by_head(csv_path: str) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for header, rows in read_total_results_tables(csv_path):
        if "test_average" not in header or len(header) < 3:
            continue
        output_head = header[2].strip()
        try:
            test_idx = header.index("test_average")
        except ValueError:
            continue

        head_scores: Dict[int, float] = {}
        for row in rows:
            if not row:
                continue
            seed_token = row[0].strip().lower()
            if seed_token == "average":
                continue
            try:
                seed = int(seed_token)
            except ValueError:
                continue
            if test_idx >= len(row):
                continue
            try:
                head_scores[seed] = float(row[test_idx])
            except ValueError:
                continue

        if head_scores:
            out[output_head] = head_scores
    return out


def best_per_seed(seed_scores_by_head: Dict[str, Dict[int, float]]) -> Dict[int, float]:
    best: Dict[int, float] = {}
    for head_scores in seed_scores_by_head.values():
        for seed, score in head_scores.items():
            if seed not in best or score > best[seed]:
                best[seed] = score
    return best


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def plot_best_compare(best_origin: Dict[str, float], best_fusion: Dict[str, float], output_dir: str) -> str:
    exps = [e for e in best_origin.keys() if e in best_fusion]
    if not exps:
        return ""

    x = list(range(len(exps)))
    width = 0.36
    origin_vals = [best_origin[e] for e in exps]
    fusion_vals = [best_fusion[e] for e in exps]
    all_vals = origin_vals + fusion_vals
    y_min = max(0.0, min(all_vals) - 0.8)
    y_max = max(all_vals) + 0.8

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x_left = [xi - width / 2 for xi in x]
    x_right = [xi + width / 2 for xi in x]
    b1 = ax.bar(x_left, origin_vals, width=width, label="优化前", color="#4C78A8")
    b2 = ax.bar(x_right, fusion_vals, width=width, label="优化后", color="#F58518")

    ax.set_title("Fusion w：最佳准确率对比")
    ax.set_ylabel("测试准确率（%）")
    ax.set_xlabel("实验设置")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=8, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.legend(title=None)

    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    out_path = os.path.join(output_dir, "图1_Fusionw_最佳准确率对比.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_delta(best_origin: Dict[str, float], best_fusion: Dict[str, float], output_dir: str) -> str:
    exps = [e for e in best_origin.keys() if e in best_fusion]
    if not exps:
        return ""

    deltas = [best_fusion[e] - best_origin[e] for e in exps]
    y_min = min(deltas) - 0.2
    y_max = max(deltas) + 0.2

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    bars = ax.bar(exps, deltas, color=["#54A24B" if d >= 0 else "#E45756" for d in deltas])
    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.6)
    ax.set_title("Fusion w：相对提升")
    ax.set_ylabel("提升（百分点）")
    ax.set_xlabel("实验设置")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.22)

    for bar, delta in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            delta + (0.03 if delta >= 0 else -0.03),
            f"{delta:+.2f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=10,
        )

    out_path = os.path.join(output_dir, "图2_Fusionw_相对提升.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_mean_std_best(
    origin_best: Dict[str, Dict[int, float]],
    fusion_best: Dict[str, Dict[int, float]],
    output_dir: str,
) -> str:
    exps = [e for e in origin_best.keys() if e in fusion_best]
    if not exps:
        return ""

    origin_means, origin_stds = [], []
    fusion_means, fusion_stds = [], []

    for exp in exps:
        seeds = sorted(set(origin_best[exp]).intersection(fusion_best[exp]))
        o_vals = [origin_best[exp][s] for s in seeds]
        f_vals = [fusion_best[exp][s] for s in seeds]
        o_mean, o_std = mean_std(o_vals)
        f_mean, f_std = mean_std(f_vals)
        origin_means.append(o_mean)
        origin_stds.append(o_std)
        fusion_means.append(f_mean)
        fusion_stds.append(f_std)

    x = list(range(len(exps)))
    width = 0.36
    x_left = [xi - width / 2 for xi in x]
    x_right = [xi + width / 2 for xi in x]

    y_all = origin_means + fusion_means
    y_min = max(0.0, min(y_all) - 1.0)
    y_max = max(y_all) + 1.0

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    b1 = ax.bar(
        x_left,
        origin_means,
        width=width,
        yerr=origin_stds,
        capsize=4,
        label="优化前",
        color="#4C78A8",
        alpha=0.95,
    )
    b2 = ax.bar(
        x_right,
        fusion_means,
        width=width,
        yerr=fusion_stds,
        capsize=4,
        label="优化后",
        color="#F58518",
        alpha=0.95,
    )

    ax.set_title("逐 seed 最佳结果：均值 ± 标准差")
    ax.set_ylabel("测试准确率（%）")
    ax.set_xlabel("实验设置")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=8, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.legend(title=None)

    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.08,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    out_path = os.path.join(output_dir, "图3_Fusionw_均值标准差.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_paired_delta(
    origin_best: Dict[str, Dict[int, float]],
    fusion_best: Dict[str, Dict[int, float]],
    output_dir: str,
) -> str:
    exps = [e for e in origin_best.keys() if e in fusion_best]
    if not exps:
        return ""

    fig, axes = plt.subplots(1, len(exps), figsize=(5.0 * len(exps), 4.4), squeeze=False)

    for idx, exp in enumerate(exps):
        ax = axes[0][idx]
        seeds = sorted(set(origin_best[exp]).intersection(fusion_best[exp]))
        if not seeds:
            continue
        o_vals = [origin_best[exp][s] for s in seeds]
        f_vals = [fusion_best[exp][s] for s in seeds]
        deltas = [f - o for o, f in zip(o_vals, f_vals)]
        avg_delta = mean(deltas) if deltas else 0.0

        for s, o_val, f_val, d in zip(seeds, o_vals, f_vals, deltas):
            color = "#54A24B" if d >= 0 else "#E45756"
            ax.plot([0, 1], [o_val, f_val], marker="o", color=color, alpha=0.8, linewidth=1.8)

        ymin = min(o_vals + f_vals) - 0.8
        ymax = max(o_vals + f_vals) + 0.8
        if ymin < 0:
            ymin = 0

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["优化前", "优化后"])
        ax.set_title(f"{exp}\n平均提升 {avg_delta:+.2f}")
        ax.grid(axis="y", linestyle="--", alpha=0.22)

    fig.suptitle("逐 seed 配对对比", y=1.0, fontsize=13)
    out_path = os.path.join(output_dir, "图4_Fusionw_逐seed配对.png")
    # Tight layout can clip the suptitle; adjust top margin explicitly.
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def plot_head_delta_heatmap(
    origin_heads: Dict[str, Dict[str, Dict[int, float]]],
    fusion_heads: Dict[str, Dict[str, Dict[int, float]]],
    output_dir: str,
) -> str:
    exps = [e for e in origin_heads.keys() if e in fusion_heads]
    if not exps:
        return ""

    shared_heads = None
    for exp in exps:
        heads = set(origin_heads[exp]).intersection(fusion_heads[exp])
        shared_heads = heads if shared_heads is None else shared_heads.intersection(heads)
    if not shared_heads:
        return ""
    heads_sorted = sorted(shared_heads)

    matrix: List[List[float]] = []
    for head in heads_sorted:
        row_vals: List[float] = []
        for exp in exps:
            o_seed_scores = origin_heads[exp].get(head, {})
            f_seed_scores = fusion_heads[exp].get(head, {})
            seeds = sorted(set(o_seed_scores).intersection(f_seed_scores))
            if not seeds:
                row_vals.append(float("nan"))
                continue
            deltas = [f_seed_scores[s] - o_seed_scores[s] for s in seeds]
            row_vals.append(mean(deltas))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(1.1 * len(exps) + 6.2, 0.38 * len(heads_sorted) + 3.0))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_title("各输出头的平均提升热力图")
    ax.set_xlabel("实验设置")
    ax.set_ylabel("输出头")
    ax.set_xticks(list(range(len(exps))))
    ax.set_xticklabels(exps, rotation=8, ha="right")
    ax.set_yticks(list(range(len(heads_sorted))))
    ax.set_yticklabels(heads_sorted)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val == val:
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("提升（百分点）")

    out_path = os.path.join(output_dir, "图5_Fusionw_头部提升热力图.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    setup_chinese_font()
    ensure_dir(args.output_dir)

    best_origin: Dict[str, float] = {}
    best_fusion: Dict[str, float] = {}
    origin_best_by_exp: Dict[str, Dict[int, float]] = {}
    fusion_best_by_exp: Dict[str, Dict[int, float]] = {}
    origin_heads_by_exp: Dict[str, Dict[str, Dict[int, float]]] = {}
    fusion_heads_by_exp: Dict[str, Dict[str, Dict[int, float]]] = {}

    for exp in args.exp_names:
        origin_csv = total_csv_path(args.results_root, args.origin_dir, args.dataset, args.backbone, exp)
        fusion_csv = total_csv_path(args.results_root, args.fusion_dir, args.dataset, args.backbone, exp)

        origin_scores = extract_average_test_scores(origin_csv)
        fusion_scores = extract_average_test_scores(fusion_csv)
        origin_heads = extract_seed_scores_by_head(origin_csv)
        fusion_heads = extract_seed_scores_by_head(fusion_csv)

        if origin_scores:
            best_origin[exp] = max(origin_scores.values())
        else:
            print(f"[警告] 未解析到优化前结果：{origin_csv}")
        if fusion_scores:
            best_fusion[exp] = max(fusion_scores.values())
        else:
            print(f"[警告] 未解析到优化后结果：{fusion_csv}")

        if origin_heads:
            origin_heads_by_exp[exp] = origin_heads
            origin_best_by_exp[exp] = best_per_seed(origin_heads)
        if fusion_heads:
            fusion_heads_by_exp[exp] = fusion_heads
            fusion_best_by_exp[exp] = best_per_seed(fusion_heads)

    paths = [
        plot_best_compare(best_origin, best_fusion, args.output_dir),
        plot_delta(best_origin, best_fusion, args.output_dir),
        plot_mean_std_best(origin_best_by_exp, fusion_best_by_exp, args.output_dir),
        plot_paired_delta(origin_best_by_exp, fusion_best_by_exp, args.output_dir),
        plot_head_delta_heatmap(origin_heads_by_exp, fusion_heads_by_exp, args.output_dir),
    ]

    print("已生成 Fusion w 可视化：")
    for p in paths:
        if p:
            print(f"- {p}")


if __name__ == "__main__":
    main()
