# Crafting Distribution Shifts（Jittor 迁移版）

本仓库为将原始 PyTorch 版 Crafting Distribution Shifts 项目迁移到 Jittor（计图）的实现。
所有核心代码已在 Jittor 版本中重写，并加入了详细中文注释，说明 PyTorch 与 Jittor 的差异与迁移逻辑。

---

## 环境配置（Environment Setup）

1) 安装 Jittor（建议 GPU 版）

- 参考官方文档安装 Jittor 与 CUDA 依赖。
- Jittor 会自动编译算子，首次运行可能较慢。

2) 安装依赖

建议在本目录下执行：

```bash
pip install jittor imgaug numpy pillow scipy scikit-image pyyaml
```

说明：
- `imgaug` 仅在 CPU/Numpy 上执行；Jittor 负责后续张量计算。
- 若使用 GPU，请确保 CUDA 与驱动版本匹配（通过 `jt.flags.use_cuda` 开关控制）。

---

## 数据准备（Data Preparation） 📁

数据文件位于：`data/PACS/`

将PACS数据集下载到 `data/PACS/` 目录下，命名为 `PACS_Original/`，
```
crafting-shifts-Jittor/
    └── data/
        └── PACS/
            └── PACS_Original/
                ├── art_painting/
                ├── cartoon/
                ├── photo/
                └── sketch/
```

生成 10 种特殊增强版本的 PACS 数据集：
```bash
python create_imgaug_datasets.py --dataset PACS
```

- 将生成多个 ImgAug 增强目录（`PACS_Imgaug_*`），并生成对应的 CSV 文件。

最终数据目录结构如下：
```
crafting-shifts-Jittor/
    └── data/
        └── PACS/
            ├── PACS_Imgaug_arithmetic/
            ├── PACS_Imgaug_artistic/
            ├── PACS_Imgaug_blur/
            ├── PACS_Imgaug_color/
            ├── PACS_Imgaug_contrast/
            ├── PACS_Imgaug_convolutional/
            ├── PACS_Imgaug_edges/
            ├── PACS_Imgaug_geometric/
            ├── PACS_Imgaug_segmentation/
            ├── PACS_Imgaug_weather/
            └── PACS_Original/
                ├── art_painting/
                ├── cartoon/
                ├── photo/
                └── sketch/
            ├── art_painting_test.csv
            ├── art_painting_train.csv
            ├── art_painting_val.csv
            ├── cartoon_test.csv
            ├── cartoon_train.csv
            ├── cartoon_val.csv
            ├── photo_test.csv
            ├── photo_train.csv
            ├── photo_val.csv
            ├── sketch_test.csv
            ├── sketch_train.csv
            └── sketch_val.csv
```

- 包含原始分割（`PACS_Original/`）和多个预生成的 ImgAug 增强目录（`PACS_Imgaug_*`）。
- CSV 文件格式为：每行 `相对路径 类别`，保持与原项目一致。

注意：本仓库不在运行时实时生成全部 ImgAug 数据，增强数据为预生成目录。

---

## 权重转换与预训练模型（Weights） 🧠

- 用于 CaffeNet/Caffe 权重转换的脚本：`convert_caffe.py`。
- 用于将 timm 的 ViT-Small PyTorch 权重转换为 Jittor 的脚本：`convert_vit_small.py`（依赖 `torch` 与 `timm`）。
- 生成的 Jittor 权重以 `.pkl` 保存，放在 `Pretrained_Models/` 中进行管理。

示例转换命令（AlexNet/Caffe）：

```bash
python convert_weights.py --src ./Pretrained_Models/alexnet_caffe.pth.tar \
                          --dst ./Pretrained_Models/alexnet_caffe_jittor.pkl
```

示例转换命令（ViT-Small from timm）：

```bash
# 从在线下载的 timm 权重直接转换（默认去掉分类 head）
python convert_vit_small.py --dst ./Pretrained_Models/vit_small_jittor.pkl

# 或使用本地 PyTorch 权重文件进行转换
python convert_vit_small.py --src ./Pretrained_Models/vit_small_patch16_224.pth \
                           --dst ./Pretrained_Models/vit_small_jittor.pkl
```

说明：`convert_vit_small.py` 会去掉分类 head（`head.*`）并保存转换后的权重为 `.pkl`。

---

## 程序执行（Execution） 🚀

### 单条命令（Single Execution）

- 使用 `method_jt.py` 作为主入口脚本，通过命令行参数指定实验配置文件（YAML）。
- 示例命令：

```bash
python method_jt.py --run experiments/yaml_PACS_imgaug_canny-all.yaml --backbone resnet18 --train_only photo --seed 0 --method_loss 1 --lr 0.00154 --epochs 300 --dataset PACS --gpu 0
```

- 汇总与可视化结果（调用 `aggregate_results.py` / `make_scatter_plots.py`）：
- 示例命令：

```bash
python aggregate_results.py --dataset PACS --backbone resnet18 --seeds 0 1 2 3 4 --main_exp_name imgaug_and_canny_training_all --cv_exp_names imgaug_and_canny_training_first imgaug_and_canny_training_second
python aggregate_results.py --dataset PACS --backbone resnet18 --seeds 0 1 2 3 4 --main_exp_name original_and_canny_training   
python aggregate_results.py --dataset PACS --backbone resnet18 --seeds 0 1 2 3 4 --main_exp_name original-only_training   
```

其余模型类型替换 `--backbone` 参数即可。

之后会在 `Results/` 目录下生成汇总结果，使用下面的命令进行可视化：

```bash
python make_scatter_plots.py --dataset PACS
```

### 批量化执行（Batch Execution）

仓库中包含若干脚本用于批量运行训练/汇总与可视化：

- `train_val.sh` : 包含多个训练/评估命令（按不同 backbone 与 seed）。
- `aggregate_visualize.sh` : 汇总并生成可视化结果的脚本（调用 `aggregate_results.py` / `make_scatter_plots.py`）。

运行整份脚本（Bash / WSL / Git Bash）：

```bash
bash train_val.sh
bash aggregate_visualize.sh
```
---

## 仓库结构（Repository structure） 📂

- `augmentations_jt.py`        : Canny/Invert/Normalize/ToTensor 等增强工具
- `utils_dataset_jt.py`        : Dataset 与 imgaug/几何增强逻辑
- `models_jt.py`               : PseudoCombiner / CaffeNet / ResNet (Jittor)
- `utils_train_inference_jt.py`: 训练/验证/搜索逻辑
- `method_jt.py`               : 主入口脚本（训练/测试）
- `convert_caffe.py` / `convert_vit_small.py` : 权重转换脚本
- `create_imgaug_datasets.py`  : 生成/管理 ImgAug 数据集工具（如需重建）
- `aggregate_results.py`, `visualize_results.py`, `make_scatter_plots.py` : 结果汇总与可视化
- `experiments/`               : 实验配置 YAML 文件（示例：`yaml_PACS_*.yaml`）
- `Pretrained_Models/`         : 预训练模型文件（.pth / .pkl）
- `data/`                      : 原始与预生成数据（通常被忽略，不提交）
- `Results/`, `Analysis_Results/`: 训练结果与分析输出（被忽略）

---

## 致谢（Acknowledgements）

感谢 [Crafting Distribution Shifts](https://github.com/NikosEfth/crafting-shifts) 提供的代码与原版实验框架。
---

## 优化方案：更稳健的 VA（val_only）选参

### 思路

原始实现会把 `val_only` 各个增强组的准确率直接取均值（`Imgaug_average_*`），再用它来选超参。  
这在组间方差较大时不够稳健：均值可能被“容易的组”主导，导致选出来的超参在测试集上不稳定。

更稳健的替代指标：
- `worst`：各组最小值（worst-group）。
- `cvar`：最差的 k 个组的均值（bottom-k mean），比最小值更平滑。

因此我们把 VA 汇总指标做成“可配置参数”，用于消融和更稳健的选参。

### 新代码如何使用

新增参数（训练端与聚合端一致）：
- `--val_only_metric`：可选 `average / worst / cvar`，可多选。
- `--val_only_cvar_k`：当选择 `cvar` 时使用，表示 bottom-k 的 k。

常用用法：

```bash
# 1) 保持原始行为（仅 average）
python method_jt.py ... --val_only_metric average

# 2) average + worst（用于消融对比）
python method_jt.py ... --val_only_metric average worst

# 3) 仅使用 CVaR（bottom-k）
python method_jt.py ... --val_only_metric cvar --val_only_cvar_k 3
```

聚合/汇总时建议保持一致：

```bash
python aggregate_results.py ... --val_only_metric average worst
python aggregate_results.py ... --val_only_metric cvar --val_only_cvar_k 3
```

### 注意事项

- 更换指标后请使用 `--search_mode new_test` 重新生成 CSV，或手动删除旧的 `Results_source_*.csv`，否则表头可能不匹配。
- 如果已经存在 `Cross-Val_*.csv`，想加入新指标也需要用新的参数重新跑一次交叉验证生成。

---

## 优化方案：更密的融合权重网格（Fusion w）+ 用 VA 自动选

### 思路

当前推理阶段只提供了 3 个融合权重（例如 25-75 / 50-50 / 75-25）。
这相当于在论文里的融合权重 `w` 上做了很粗的网格搜索，可能错过更优的权重。

优化思路：
- 在推理阶段生成更密的融合输出（例如 10-90, 20-80, ..., 90-10）。
- 继续用现有的 VA / val_only 机制自动选择最优输出头（无需改训练）。

这样基本不增加训练成本，但能更完整地覆盖融合权重空间。

### 代码改动位置

- 融合输出生成（推理阶段）：
  - `models_jt.py` 中 `PseudoCombiner.execute`
- 输出头名称同步（避免 CSV 表头对不上）：
  - `utils_dataset_jt.py` 中 `set_dataloaders` 的 `output_names_val`
- 参数入口（用于消融开关）：
  - `method_jt.py` 中 `--fusion_weights`

### 新代码如何使用

这个优化现在由超参数控制：
- `--fusion_weights`：一组 `w0`，融合时用 `w1 = 1 - w0`。

示例：

```bash
# 1) 使用更密的融合网格（当前默认）
python method_jt.py ... --fusion_weights 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.9 --search_mode new_test

# 2) 退回原始的三档融合（方便消融）
python method_jt.py ... --fusion_weights 0.25 0.5 0.75 --search_mode new_test
```

聚合时无需额外参数，但建议在更换 `fusion_weights` 后重新生成 CSV：

```bash
python aggregate_results.py ...
```

### 注意事项

- 由于输出头集合会随 `fusion_weights` 变化，旧的 `Results_source_*.csv` / `Cross-Val_*.csv` 可能与新表头不一致，建议用 `--search_mode new_test` 重跑。
- 训练逻辑不变：这些融合输出只在 eval 生成，不参与训练 loss。
# 可视化使用说明

## 结果可视化（visualize_results.py）

`visualize_results.py` 会自动扫描 `Results/` 下各模型目录中的实验结果 CSV，
并在 `Analysis_Results/` 下生成单实验图、单模型汇总图，以及跨模型/跨实验的全局汇总图。

直接运行：

```bash
python visualize_results.py
```

默认约定：
- 结果目录：`Results/PACS_caffenet`、`Results/PACS_resnet18`、`Results/PACS_vit_small`
- CSV 文件：`Results_source_photo_seed_0.csv`（若存在子目录，会尝试 `imgaug_and_canny_training_all/Results_source_photo_seed_0.csv`）
- 输出目录：`Analysis_Results/`

如果你的结果目录或 CSV 命名不同，请在 `visualize_results.py` 顶部的 `base_path` 与 `model_dirs` 中进行相应修改。

## 参数选择对比（visualize_VA_param.py）

```bash
python visualize_VA_param.py --dataset PACS --backbone resnet18 --results_root Results --variants origin worst cvar --exp_names imgaug_and_canny_training_all original_and_canny_training original-only_training
```

输出目录：`Analysis_Results/VA_param`

## Fusion w 优化前后对比（visualize_fusion_w_cn.py）

```bash
python visualize_fusion_w.py --dataset PACS --backbone resnet18 --results_root Results --origin_dir Results_origin --fusion_dir Results_Fusion_w --exp_names imgaug_and_canny_training_all original_and_canny_training original-only_training
```

输出目录：`Analysis_Results/fusion_w`
