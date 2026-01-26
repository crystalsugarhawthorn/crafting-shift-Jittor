import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math
import matplotlib.font_manager as fm


def find_chinese_font():
    """Return the name of an installed font that likely supports Chinese characters, or None."""
    candidates = [
        'noto', 'wqy', 'simhei', 'msyh', 'simsun', 'source han', 'pingfang', 'ar pl uming', 'microsoft yahei', 'noto sans cjk'
    ]
    for f in fm.fontManager.ttflist:
        try:
            name = f.name.lower()
        except Exception:
            continue
        for kw in candidates:
            if kw in name:
                return f.name
    return None

# ==========================================
# 1. 学术风格设置 (Academic Style Settings)
# ==========================================
def set_academic_style():
    # 先设置 Seaborn 样式，避免覆盖后续的字体设置
    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # 设置中文字体，优先使用 Microsoft YaHei (Windows默认)，备选 SimHei, SimSun
    # 注意：必须在 sns.set_style 之后设置，否则会被覆盖
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8

    # 优先使用系统中可用的中文字体以避免中文缺字问题
    ch_font = find_chinese_font()
    if ch_font:
        plt.rcParams['font.sans-serif'] = [ch_font] + plt.rcParams.get('font.sans-serif', [])
        print(f"Using Chinese font: {ch_font}")
    else:
        # 尝试从常见系统路径加载字体文件（Noto/WenQuanYi/Arphic/SimHei 等）
        def try_load_fallback_chinese_font():
            candidates = [
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf',
                '/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttf',
                '/usr/share/fonts/truetype/arphic/uming.ttc',
                '/usr/share/fonts/truetype/arphic/ukai.ttc',
                '/usr/share/fonts/truetype/microsoft/SimHei.ttf',
                os.path.expanduser('~/.local/share/fonts/NotoSansCJK-Regular.ttc'),
            ]
            for p in candidates:
                if os.path.exists(p):
                    try:
                        fm.fontManager.addfont(p)
                        prop = fm.FontProperties(fname=p)
                        name = prop.get_name()
                        plt.rcParams['font.sans-serif'] = [name] + plt.rcParams.get('font.sans-serif', [])
                        plt.rcParams['font.family'] = 'sans-serif'
                        print(f"Loaded Chinese font from file: {p} -> using {name}")
                        return name
                    except Exception as e:
                        print(f"Failed to load font file {p}: {e}")
            return None

        loaded = try_load_fallback_chinese_font()
        if not loaded:
            print("Warning: No Chinese font found; Chinese glyphs may be missing. Consider installing Noto Sans CJK or SimHei (apt install fonts-noto-cjk fonts-wqy-microhei).")

# ==========================================
# 2. 数据加载与预处理 (Data Loading & Preprocessing)
# ==========================================
def load_and_process_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None, None, None

    df = pd.read_csv(filepath)
    
    # 提取一行数据作为字典
    data = df.iloc[0].to_dict()
    
    # 动态检测策略后缀（例如: ImgAug, canny, 25-75, normal 等）
    cols = list(df.columns)
    suffixes = set()
    # 常见前缀集合，覆盖 output_, test_, Imgaug_average_, Imgaug_{aug}_
    imgaug_prefixes = ['Imgaug_average_', 'Imgaug_arithmetic_', 'Imgaug_artistic_', 'Imgaug_blur_', 
                       'Imgaug_color_', 'Imgaug_contrast_', 'Imgaug_convolutional_', 'Imgaug_edges_',
                       'Imgaug_geometric_', 'Imgaug_segmentation_', 'Imgaug_weather_']

    for c in cols:
        if c.startswith('output_'):
            suffixes.add(c[len('output_'):])
        elif c.startswith('test_'):
            suffixes.add(c[len('test_'):])
        else:
            for p in imgaug_prefixes:
                if c.startswith(p):
                    suffixes.add(c[len(p):])
                    break

    # 如果没有检测到任何常规策略后缀，回退到默认策略集（保证向后兼容）
    if not suffixes:
        strategies = ['output_ImgAug', 'output_canny', 'output_25-75', 'output_50-50', 'output_75-25']
        strategy_labels = [
            'RGB (纹理偏好)', 
            '形状 (Canny)', 
            '25% 纹理 + 75% 形状', 
            '50% 纹理 + 50% 形状', 
            '75% 纹理 + 25% 形状'
        ]
        strategy_map = dict(zip(strategies, strategy_labels))
        return data, strategies, strategy_map

    # 构造策略列表，例如 'output_normal'
    sorted_suffixes = sorted(list(suffixes))
    strategies = [f'output_{s}' for s in sorted_suffixes]

    # 构造中文标签映射
    def label_for_suffix(s):
        if s in ['canny']:
            return '形状 (Canny)'
        if s in ['75-25']:
            return '75% 纹理 + 25% 形状'
        if s in ['50-50']:
            return '50% 纹理 + 50% 形状'
        if s in ['25-75']:
            return '25% 纹理 + 75% 形状'
        if s.lower() in ['normal', 'orig', 'original']:
            return 'Baseline (无特殊增强)'
        if s.lower() in ['imgaug', 'img_aug', 'imgaug_average']:
            return 'RGB (纹理偏好)'
        # 默认使用后缀原样
        return s

    strategy_labels = [label_for_suffix(s) for s in sorted_suffixes]
    strategy_map = dict(zip(strategies, strategy_labels))

    return data, strategies, strategy_map


def has_any_plot_data(data, strategies):
    """检查 data 中是否包含任何将用于绘图的关键字段，若都不存在则返回 False。"""
    if data is None:
        return False

    keys = []
    # 常见域/策略键
    keys += ['output_' + s.split('_')[1] for s in strategies]
    keys += ['Imgaug_average_' + s for s in strategies]
    keys += ['test_' + s for s in strategies]

    # 增强类型键
    aug_types_en = [
        'arithmetic', 'artistic', 'blur', 'color', 'contrast', 
        'convolutional', 'edges', 'geometric', 'segmentation', 'weather'
    ]
    for aug in aug_types_en:
        for s in strategies:
            keys.append(f"Imgaug_{aug}_{s}")

    for k in keys:
        if data.get(k) is not None:
            return True
    return False

# ==========================================
# 3. 单模型分析函数 (Individual Model Analysis)
# ==========================================

def plot_domain_performance(data, strategies, strategy_map, output_dir, model_name):
    """
    图1: 跨域性能对比 (分组柱状图)
    """
    domains = {
        '源域 (照片)': ['output_' + s.split('_')[1] for s in strategies],
        '增强验证集 (平均)': ['Imgaug_average_' + s for s in strategies],
        '目标域 (平均)': ['test_' + s for s in strategies],
        '目标域 (艺术画)': ['art_painting_' + s for s in strategies],
        '目标域 (卡通)': ['cartoon_' + s for s in strategies],
        '目标域 (素描)': ['sketch_' + s for s in strategies]
    }
    
    plot_data = []
    for domain_name, keys in domains.items():
        for i, key in enumerate(keys):
            val = data.get(key)
            if val is not None:
                plot_data.append({
                    '域': domain_name,
                    '策略': strategy_map[strategies[i]],
                    '准确率': val
                })
            
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(16, 9)) # 增加高度以容纳底部注释
    ax = sns.barplot(x='域', y='准确率', hue='策略', data=df_plot, alpha=0.9)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8, rotation=90)
        
    plt.title(f'{model_name}: 跨域性能对比', pad=20)
    plt.ylim(0.0, 1.15)
    plt.ylabel('准确率')
    plt.xlabel('')
    plt.legend(loc='upper right', frameon=True, framealpha=0.9, ncol=3)
    
    # 添加底部注释
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：该图展示了模型在源域、增强验证集以及各个目标域上的准确率表现，对比了不同推理策略的效果。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, f'{model_name}_1_domain_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_augmentation_heatmap(data, strategies, strategy_map, output_dir, model_name):
    """
    图2: 增强类型鲁棒性热力图
    """
    # 增强类型中文映射
    aug_types_en = [
        'arithmetic', 'artistic', 'blur', 'color', 'contrast', 
        'convolutional', 'edges', 'geometric', 'segmentation', 'weather'
    ]
    aug_types_cn = [
        '算术运算', '艺术风格', '模糊', '颜色变换', '对比度', 
        '卷积滤波', '边缘提取', '几何变换', '分割', '天气'
    ]
    
    # 过滤掉在所有增强类型上都为缺失或 0 的策略列
    filtered_strategies = []
    for strat in strategies:
        # 若某一增强类型下有非 0/非 None 的值，则保留该策略
        has_nonzero = any(
            (data.get(f"Imgaug_{aug}_{strat}", None) is not None and data.get(f"Imgaug_{aug}_{strat}") != 0)
            for aug in aug_types_en
        )
        if has_nonzero:
            filtered_strategies.append(strat)

    if not filtered_strategies:
        print("Warning: No valid Imgaug strategy data found for heatmap; skipping heatmap.")
        return

    heatmap_data = []
    for aug in aug_types_en:
        row = []
        for strat in filtered_strategies:
            key = f"Imgaug_{aug}_{strat}"
            row.append(data.get(key, 0))
        heatmap_data.append(row)

    # 使用过滤后的策略映射为列标题（使用中文 label，如果缺失则退回策略名）
    col_labels = [strategy_map.get(s, s) for s in filtered_strategies]
    df_heatmap = pd.DataFrame(heatmap_data, index=aug_types_cn, columns=col_labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': '准确率'})
    
    plt.title(f'{model_name}: 验证集增强鲁棒性热力图', pad=20)
    plt.ylabel('增强类型')
    plt.xlabel('推理策略')
    plt.xticks(rotation=45, ha='right')
    
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：颜色越深代表准确率越高。该图展示了模型在面对不同类型的图像扰动时，各策略的鲁棒性表现。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, f'{model_name}_2_augmentation_robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_strategy_sensitivity(data, strategies, strategy_map, output_dir, model_name):
    """
    图3: 策略敏感度分析 (折线图)
    """
    ordered_strategies = ['output_ImgAug', 'output_75-25', 'output_50-50', 'output_25-75', 'output_canny']
    ordered_labels = [
        'RGB (100%)', 
        '75% 纹理 + 25% 形状', 
        '50% 纹理 + 50% 形状', 
        '25% 纹理 + 75% 形状', 
        '形状 (100%)'
    ]
    
    metrics = {
        '源域验证 (照片)': ['output_' + s.split('_')[1] for s in ordered_strategies],
        '增强验证 (平均)': ['Imgaug_average_' + s for s in ordered_strategies],
        '目标域测试 (平均)': ['test_' + s for s in ordered_strategies]
    }
    
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^']
    for i, (metric_name, keys) in enumerate(metrics.items()):
        values = [data.get(key) for key in keys]
        plt.plot(ordered_labels, values, marker=markers[i], label=metric_name, linestyle='-')
        
    plt.title(f'{model_name}: 形状偏好对泛化性的影响', pad=20)
    plt.ylabel('准确率')
    plt.xlabel('推理策略 (纹理 vs 形状权重)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：该图展示了随着形状权重逐渐增加（从左至右），模型在源域、增强域和目标域上的性能变化趋势。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, f'{model_name}_3_strategy_sensitivity.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_augmentations(data, strategies, strategy_map, output_dir, model_name):
    """
    图4: 增强类型雷达图 (Radar Chart)
    展示所有策略在各个增强类型上的表现
    """
    aug_types_en = [
        'arithmetic', 'artistic', 'blur', 'color', 'contrast', 
        'convolutional', 'edges', 'geometric', 'segmentation', 'weather'
    ]
    aug_types_cn = [
        '算术', '艺术', '模糊', '颜色', '对比度', 
        '卷积', '边缘', '几何', '分割', '天气'
    ]
    
    # 选择要展示的策略 - 用户要求全部展示
    selected_strategies = strategies 
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(aug_types_en), endpoint=False).tolist()
    angles += angles[:1] # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for strat in selected_strategies:
        values = []
        for aug in aug_types_en:
            key = f"Imgaug_{aug}_{strat}"
            values.append(data.get(key, 0))
        values += values[:1] # 闭合
        
        ax.plot(angles, values, linewidth=2, label=strategy_map[strat])
        ax.fill(angles, values, alpha=0.05)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(aug_types_cn, fontsize=12)
    ax.set_ylim(0, 1.0)
    
    plt.title(f'{model_name}: 鲁棒性分布雷达图 (全策略)', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.subplots_adjust(bottom=0.1)
    plt.figtext(0.5, 0.02, "注：该雷达图展示了该模型下所有策略在10种不同图像增强类型上的性能分布。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, f'{model_name}_4_radar_robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. 综合对比分析函数 (Combined Analysis)
# ==========================================

def plot_combined_test_accuracy(all_data, output_dir):
    """
    综合图1: 不同模型在 Target Domain (Avg) 上的表现对比
    包含所有策略
    """
    # 动态提取所有实验中存在的策略后缀
    all_suffixes = set()
    for data in all_data.values():
        if data is None: continue
        for key in data.keys():
            if key.startswith('test_output_'):
                all_suffixes.add(key[len('test_output_'):])
    
    # 如果没有检测到，回退到默认
    if not all_suffixes:
        all_suffixes = {'ImgAug', '75-25', '50-50', '25-75', 'canny'}
    
    # 生成策略列表和标签
    def label_for_suffix(s):
        if s == 'canny': return 'Shape'
        if s == 'ImgAug': return 'RGB'
        if s in ['normal', 'orig']: return 'Baseline'
        return s
    
    sorted_suffixes = sorted(list(all_suffixes))
    strategies = [f'output_{s}' for s in sorted_suffixes]
    strategy_labels = [label_for_suffix(s) for s in sorted_suffixes]
    
    plot_data = []
    
    for model_name, data in all_data.items():
        if data is None: continue
        for strat, label in zip(strategies, strategy_labels):
            key = 'test_' + strat
            val = data.get(key)
            if val is not None:
                plot_data.append({
                    '模型': model_name,
                    '策略': label,
                    '准确率': val
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(18, 8))
    ax = sns.barplot(x='模型', y='准确率', hue='策略', data=df_plot)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=7, rotation=90)
    
    plt.xticks(rotation=15, ha='right')
        
    plt.title('综合分析: 目标域平均准确率对比 (全策略)', pad=20)
    plt.ylim(0, 1.15) # 增加高度以容纳标签
    plt.ylabel('平均测试准确率')
    plt.legend(loc='upper right', ncol=5)
    
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：对比了不同模型在所有目标域上的平均测试准确率，包含所有混合策略。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, 'Combined_1_Test_Accuracy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_robustness(all_data, output_dir):
    """
    综合图2: 不同模型在 Augmentation Validation (Avg) 上的表现对比
    包含所有策略
    """
    # 动态提取所有实验中存在的策略后缀
    all_suffixes = set()
    for data in all_data.values():
        if data is None: continue
        for key in data.keys():
            if key.startswith('Imgaug_average_output_'):
                all_suffixes.add(key[len('Imgaug_average_output_'):])
    
    if not all_suffixes:
        all_suffixes = {'ImgAug', '75-25', '50-50', '25-75', 'canny'}
    
    def label_for_suffix(s):
        if s == 'canny': return 'Shape'
        if s == 'ImgAug': return 'RGB'
        if s in ['normal', 'orig']: return 'Baseline'
        return s
    
    sorted_suffixes = sorted(list(all_suffixes))
    strategies = [f'output_{s}' for s in sorted_suffixes]
    strategy_labels = [label_for_suffix(s) for s in sorted_suffixes]
    
    plot_data = []
    
    for model_name, data in all_data.items():
        if data is None: continue
        for strat, label in zip(strategies, strategy_labels):
            key = 'Imgaug_average_' + strat
            val = data.get(key)
            if val is not None:
                plot_data.append({
                    '模型': model_name,
                    '策略': label,
                    '准确率': val
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(18, 8))
    ax = sns.barplot(x='模型', y='准确率', hue='策略', data=df_plot, palette='viridis')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=7, rotation=90)
    
    plt.xticks(rotation=15, ha='right')
        
    plt.title('综合分析: 增强验证集鲁棒性对比 (全策略)', pad=20)
    plt.ylim(0, 1.15)
    plt.ylabel('平均鲁棒性准确率')
    plt.legend(loc='upper right', ncol=5)
    
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：对比了不同模型在增强验证集上的平均鲁棒性准确率，包含所有混合策略。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, 'Combined_2_Robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_strategy_curve(all_data, output_dir):
    """
    综合图3: 所有模型的策略敏感度曲线对比 (Target Test Avg)
    """
    # 动态提取所有实验中存在的策略后缀
    all_suffixes = set()
    baseline_suffixes = set()
    for data in all_data.values():
        if data is None: continue
        for key in data.keys():
            if key.startswith('test_output_'):
                suffix = key[len('test_output_'):]
                if suffix in ['normal', 'orig']:
                    baseline_suffixes.add(suffix)
                else:
                    all_suffixes.add(suffix)
    
    # 定义固定的策略顺序（不含 Baseline）
    standard_order = ['ImgAug', '75-25', '50-50', '25-75', 'canny']
    ordered_suffixes = [s for s in standard_order if s in all_suffixes]
    
    # 如果有检测到的策略不在标准列表中，追加到末尾
    for s in sorted(all_suffixes):
        if s not in ordered_suffixes:
            ordered_suffixes.append(s)
    
    def label_for_suffix(s):
        if s == 'canny': return 'Shape'
        if s == 'ImgAug': return 'RGB'
        if s in ['normal', 'orig']: return 'Baseline'
        return s
    
    ordered_strategies = [f'output_{s}' for s in ordered_suffixes]
    ordered_labels = [label_for_suffix(s) for s in ordered_suffixes]
    
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.tab10(range(10))
    
    for i, (model_name, data) in enumerate(all_data.items()):
        if data is None: continue
        values = []
        x_positions = []
        
        # 绘制标准策略曲线（不含 Baseline）
        for idx, strat in enumerate(ordered_strategies):
            key = 'test_' + strat
            val = data.get(key)
            if val is not None:
                values.append(val)
                x_positions.append(idx)
            
        if len(values) > 0:
            # 绘制曲线
            plt.plot(x_positions, values, marker=markers[i % len(markers)], 
                    label=model_name, linewidth=2.5, color=colors[i % len(colors)])
        
        # 单独处理 Baseline：在图表右侧绘制散点
        for baseline_suffix in baseline_suffixes:
            baseline_key = f'test_output_{baseline_suffix}'
            baseline_val = data.get(baseline_key)
            if baseline_val is not None:
                # Baseline 显示在最右侧，位置为曲线结束后 + 1.5
                baseline_x = len(ordered_strategies) + 0.5
                plt.scatter([baseline_x], [baseline_val], marker=markers[i % len(markers)], 
                           s=150, color=colors[i % len(colors)], zorder=3, edgecolors='black', linewidths=1.5)
        
    plt.title('综合分析: 目标域策略敏感度曲线', pad=20)
    plt.ylabel('测试准确率')
    plt.xlabel('推理策略')
    
    # 设置 x 轴刻度：标准策略 + Baseline（如果存在）
    all_x_positions = list(range(len(ordered_labels)))
    all_x_labels = ordered_labels.copy()
    
    if baseline_suffixes:
        baseline_x = len(ordered_strategies) + 0.5
        all_x_positions.append(baseline_x)
        all_x_labels.append('Baseline')
    
    plt.xticks(all_x_positions, all_x_labels, rotation=30, ha='right', fontsize=10)
    plt.xlim(-0.5, max(all_x_positions) + 0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：展示了各模型在目标域上的性能随策略（从RGB到Shape）变化的趋势。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, 'Combined_3_Strategy_Curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_domain_breakdown(all_data, output_dir):
    """
    综合图4: 各个 Target Domain (Art, Cartoon, Sketch) 的详细对比
    """
    domains = ['art_painting', 'cartoon', 'sketch']
    
    # 动态提取所有实验中存在的策略后缀
    all_suffixes = set()
    for data in all_data.values():
        if data is None: continue
        for key in data.keys():
            for domain in domains:
                if key.startswith(f'{domain}_output_'):
                    all_suffixes.add(key[len(f'{domain}_output_'):])
    
    if not all_suffixes:
        all_suffixes = {'ImgAug', '75-25', '50-50', '25-75', 'canny'}
    
    sorted_suffixes = sorted(list(all_suffixes))
    strategies = [f'output_{s}' for s in sorted_suffixes]
    
    plot_data = []
    
    for model_name, data in all_data.items():
        if data is None: continue
        for domain in domains:
            for strat in strategies:
                key = f"{domain}_{strat}"
                val = data.get(key)
                
                # 动态标签映射
                suffix = strat.replace('output_', '')
                if suffix == 'ImgAug': label = 'RGB'
                elif suffix == 'canny': label = 'Shape'
                elif suffix in ['normal', 'orig']: label = 'Baseline'
                else: label = suffix
                
                if val is not None:
                    plot_data.append({
                        '模型': model_name,
                        '域': domain.capitalize(),
                        '策略': label,
                        '准确率': val
                    })
                    
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(16, 8))
    
    g = sns.catplot(
        data=df_plot, kind="bar",
        x="域", y="准确率", hue="策略", col="模型",
        palette="muted", alpha=.9, height=6, aspect=1.1, legend_out=True
    )
    g.set_xticklabels(rotation=20, ha='right')
    g.despine(left=True)
    g.set_axis_labels("", "准确率")
    g.legend.set_title("策略")
    g.fig.suptitle('综合分析: 各目标域详细性能对比 (全策略)', y=1.05)
    
    # 由于 catplot 是 figure-level，添加 figtext 需要小心
    # g.fig 是 matplotlib figure 对象
    g.fig.subplots_adjust(bottom=0.15)
    g.fig.text(0.5, 0.02, "注：详细展示了每个模型在艺术画、卡通、素描三个域上，所有5种策略的具体表现。", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, 'Combined_4_Domain_Breakdown.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_radar_robustness(all_data, output_dir):
    """
    综合图5: 综合雷达图，对比各模型在最佳策略（假设为50-50）下的鲁棒性分布
    """
    aug_types_en = [
        'arithmetic', 'artistic', 'blur', 'color', 'contrast', 
        'convolutional', 'edges', 'geometric', 'segmentation', 'weather'
    ]
    aug_types_cn = [
        '算术', '艺术', '模糊', '颜色', '对比度', 
        '卷积', '边缘', '几何', '分割', '天气'
    ]
    strat = 'output_50-50' # 使用 Ensemble 策略进行对比
    
    angles = np.linspace(0, 2*np.pi, len(aug_types_en), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for model_name, data in all_data.items():
        if data is None: continue
        values = []
        for aug in aug_types_en:
            key = f"Imgaug_{aug}_{strat}"
            values.append(data.get(key, 0))
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.05)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(aug_types_cn, fontsize=12)
    ax.set_ylim(0, 1.0)
    
    plt.title('综合分析: 鲁棒性分布对比 (50-50 策略)', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.subplots_adjust(bottom=0.1)
    plt.figtext(0.5, 0.02, "注：对比了三个模型在 50-50 混合策略下，对不同类型图像扰动的抵抗能力。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, 'Combined_5_Radar_Robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_scatter_robustness_vs_target(all_data, output_dir):
    """
    综合图6: 散点图 - 鲁棒性 vs 目标域准确率
    """
    # 动态提取所有实验中存在的策略后缀
    all_suffixes = set()
    for data in all_data.values():
        if data is None: continue
        for key in data.keys():
            if key.startswith('test_output_'):
                all_suffixes.add(key[len('test_output_'):])
    
    if not all_suffixes:
        all_suffixes = {'ImgAug', '75-25', '50-50', '25-75', 'canny'}
    
    def label_for_suffix(s):
        if s == 'canny': return 'Shape'
        if s == 'ImgAug': return 'RGB'
        if s in ['normal', 'orig']: return 'Baseline'
        return s
    
    sorted_suffixes = sorted(list(all_suffixes))
    strategies = [f'output_{s}' for s in sorted_suffixes]
    strategy_labels = [label_for_suffix(s) for s in sorted_suffixes]
    
    plot_data = []
    
    for model_name, data in all_data.items():
        if data is None: continue
        for strat, label in zip(strategies, strategy_labels):
            rob_key = 'Imgaug_average_' + strat
            target_key = 'test_' + strat
            
            rob_val = data.get(rob_key)
            target_val = data.get(target_key)
            
            if rob_val is not None and target_val is not None:
                plot_data.append({
                    '模型': model_name,
                    '策略': label,
                    '鲁棒性 (平均)': rob_val,
                    '目标域准确率 (平均)': target_val
                })
                
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df_plot, 
        x='鲁棒性 (平均)', 
        y='目标域准确率 (平均)', 
        hue='模型', 
        style='策略', 
        s=200,
        alpha=0.8
    )
    
    # 添加对角线参考
    plt.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.5)
    
    plt.title('综合分析: 鲁棒性 vs 目标域准确率权衡', pad=20)
    plt.xlim(0.4, 1.0)
    plt.ylim(0.4, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：展示了鲁棒性与目标域泛化能力之间的关系。越靠近右上角代表模型性能越好。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, 'Combined_6_Scatter_Tradeoff.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_core_results_comparison(model_dirs, output_dir):
    """
    核心结果图: 对比 Baseline (Original-only Training + Standard Val) 
    与 Proposed Method (ImgAug+Canny Training + Cross-val Val)。
    """
    print("\nGenerating Core Result: Baseline vs Proposed Method...")
    
    results = []
    
    # 定义实验文件夹名称
    exp_baseline = 'original-only_training'
    exp_proposed = 'imgaug_and_canny_training_all'
    
    for model_name, model_dir in model_dirs.items():
        print(f"  Processing model: {model_name}")
        
        # 1. 获取 Baseline 结果
        # 路径: .../original-only_training/Scatter_Standard.csv
        path_baseline = os.path.join(model_dir, exp_baseline, "Scatter_Standard.csv")
        score_baseline = None
        
        if os.path.exists(path_baseline):
            try:
                df = pd.read_csv(path_baseline)
                # 假设第二列是 Test 准确率
                if not df.empty and df.shape[1] >= 2:
                    score_baseline = df.iloc[:, 1].mean()
            except Exception as e:
                print(f"    Error reading baseline {path_baseline}: {e}")
        else:
            print(f"    Baseline file not found: {path_baseline}")

        # 2. 获取 Proposed 结果
        # 路径: .../imgaug_and_canny_training_all/Scatter_Cross_val_Imgaug_average.csv
        path_proposed = os.path.join(model_dir, exp_proposed, "Scatter_Cross_val_Imgaug_average.csv")
        score_proposed = None
        
        if os.path.exists(path_proposed):
            try:
                df = pd.read_csv(path_proposed)
                if not df.empty and df.shape[1] >= 2:
                    score_proposed = df.iloc[:, 1].mean()
            except Exception as e:
                print(f"    Error reading proposed {path_proposed}: {e}")
        else:
            print(f"    Proposed file not found: {path_proposed}")
            
        # 添加到结果列表
        if score_baseline is not None:
            results.append({
                '模型': model_name,
                '方法': 'Baseline (original only)',
                '测试集准确率': score_baseline
            })
        
        if score_proposed is not None:
            results.append({
                '模型': model_name,
                '方法': 'Optimized (imgaug and canny training all)',
                '测试集准确率': score_proposed
            })

    if not results:
        print("  No data found for core comparison. Skipping.")
        return

    df_plot = pd.DataFrame(results)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    palette = {"Baseline (original only)": "#b1bebe", "Optimized (imgaug and canny training all)": "#3c83e7"}
    
    ax = sns.barplot(x='模型', y='测试集准确率', hue='方法', data=df_plot, palette=palette)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10, fontweight='bold')
        
    plt.title('核心结果: Baseline vs 本文方法', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('平均测试集准确率 (%)', fontsize=14)
    plt.xlabel('骨干网络', fontsize=14)
    
    # 动态调整 Y 轴范围
    if not df_plot.empty:
        y_max = df_plot['测试集准确率'].max()
        y_min = df_plot['测试集准确率'].min()
        # 如果数据是 0-1 范围
        if y_max <= 1.0:
             plt.ylim(0, 1.1)
        else:
             # 稍微留点空间
             plt.ylim(max(0, y_min - 10), y_max + 5)

    plt.legend(loc='upper left', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.subplots_adjust(bottom=0.15)
    plt.figtext(0.5, 0.02, "注：对比了 Baseline (仅使用原始数据训练，标准验证) 与 本文方法 (使用增强数据训练，交叉验证) 的性能。", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    save_path = os.path.join(output_dir, 'Core_Result_Baseline_vs_Proposed.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Core result plot saved to: {save_path}")

# ==========================================
# 5. 主执行函数 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 扩展：遍历每个模型目录下的所有实验子文件夹，分别绘制每个实验的可视化，并生成模型级与全局汇总
    base_path = r"./Results"
    model_dirs = {
        "CaffeNet": os.path.join(base_path, "PACS_caffenet"),
        "ResNet18": os.path.join(base_path, "PACS_resnet18"),
        "ViT-Small": os.path.join(base_path, "PACS_vit_small")
    }

    output_dir = r"./Analysis_Results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Output directory: {output_dir}")
    set_academic_style()

    # 全局收集：用于跨模型与跨实验的综合对比
    global_all_data = {}

    # 遍历每个模型目录
    for model_name, model_dir in model_dirs.items():
        print(f"\nScanning model directory: {model_name} -> {model_dir}")
        model_out_base = os.path.join(output_dir, model_name)
        if not os.path.exists(model_out_base):
            os.makedirs(model_out_base)

        # 收集该模型下各实验的数据
        experiments_data = {}

        if not os.path.exists(model_dir):
            print(f"Warning: 模型目录不存在: {model_dir}. 跳过 {model_name}。")
            continue

        # 列出模型目录下所有一级子文件夹（实验名）
        for entry in sorted(os.listdir(model_dir)):
            exp_path = os.path.join(model_dir, entry)
            if not os.path.isdir(exp_path):
                continue

            # 查找常见聚合结果文件名（Results_source_photo_seed_0.csv）
            csv_candidate = os.path.join(exp_path, "Results_source_photo_seed_0.csv")
            if not os.path.exists(csv_candidate):
                # 有时结果在子子目录（如 imgaug_and_canny_training_all/Results_source...）
                csv_candidate = os.path.join(exp_path, "imgaug_and_canny_training_all", "Results_source_photo_seed_0.csv")

            if os.path.exists(csv_candidate):
                print(f" Found experiment '{entry}' CSV: {csv_candidate}")
                data, strategies, strategy_map = load_and_process_data(csv_candidate)
                if data is None:
                    print(f"  Warning: 无法读取 CSV: {csv_candidate}")
                    continue

                experiments_data[entry] = data

                # 为该实验创建输出子目录
                exp_out_dir = os.path.join(model_out_base, entry)
                if not os.path.exists(exp_out_dir):
                    os.makedirs(exp_out_dir)
                # 如果数据中没有任何绘图所需字段，则跳过绘图
                if not has_any_plot_data(data, strategies):
                    print(f"  Warning: 实验 '{entry}' 中没有可绘制的数据字段，已跳过。")
                else:
                    # 调用已有绘图函数，保存到实验子文件夹
                    plot_domain_performance(data, strategies, strategy_map, exp_out_dir, f"{model_name}_{entry}")
                    plot_augmentation_heatmap(data, strategies, strategy_map, exp_out_dir, f"{model_name}_{entry}")
                    plot_strategy_sensitivity(data, strategies, strategy_map, exp_out_dir, f"{model_name}_{entry}")
                    plot_radar_augmentations(data, strategies, strategy_map, exp_out_dir, f"{model_name}_{entry}")
            else:
                # 如果没有 CSV，则跳过
                # print(f" No results CSV in {exp_path}")
                continue

        # 如果该模型至少有一个实验数据，生成模型级合并对比（以实验为比较对象）
        if experiments_data:
            print(f" Generating per-model combined plots for {model_name} ({len(experiments_data)} experiments)")
            # 用实验名作为键，重用现有的综合绘图函数（它们接受 {name: data} 格式）
            per_model_combined_dir = os.path.join(model_out_base, "Combined_Experiments")
            if not os.path.exists(per_model_combined_dir):
                os.makedirs(per_model_combined_dir)

            # 调用综合函数，但为了避免覆盖全局文件，保存时会放到 per_model_combined_dir
            # 这些综合函数会使用传入的 output_dir 来存储文件
            plot_combined_test_accuracy(experiments_data, per_model_combined_dir)
            plot_combined_robustness(experiments_data, per_model_combined_dir)
            plot_combined_strategy_curve(experiments_data, per_model_combined_dir)
            plot_combined_domain_breakdown(experiments_data, per_model_combined_dir)
            plot_combined_radar_robustness(experiments_data, per_model_combined_dir)
            plot_combined_scatter_robustness_vs_target(experiments_data, per_model_combined_dir)

            # 将每个实验最佳/重要指标收集到全局字典，键名使用 '模型|实验' 以便全局对比
            for exp_name, d in experiments_data.items():
                global_key = f"{model_name}|{exp_name}"
                global_all_data[global_key] = d

        else:
            print(f" No experiments with CSV found for model {model_name}.")

    # 生成跨模型与跨实验的全局综合分析（如果有数据）
    print("\nGenerating Global Combined Analysis (across models and experiments)...")
    if global_all_data:
        global_out_dir = os.path.join(output_dir, "Global_Combined")
        if not os.path.exists(global_out_dir):
            os.makedirs(global_out_dir)

        # 这些图会以键（'模型|实验'）作为标签，便于交叉对比
        plot_combined_test_accuracy(global_all_data, global_out_dir)
        plot_combined_robustness(global_all_data, global_out_dir)
        plot_combined_strategy_curve(global_all_data, global_out_dir)
        plot_combined_domain_breakdown(global_all_data, global_out_dir)
        plot_combined_radar_robustness(global_all_data, global_out_dir)
        plot_combined_scatter_robustness_vs_target(global_all_data, global_out_dir)
        
        # 生成核心结果对比图
        plot_core_results_comparison(model_dirs, global_out_dir)

        print(f" Global combined plots saved to: {global_out_dir}")
    else:
        print(" No global data found to plot.")

    print("\nAll visualizations complete! Check the 'Analysis_Results' directory.")
