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