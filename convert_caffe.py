import os
import torch
import jittor as jt
from collections import OrderedDict
import models_jt as models


# ===============================
# PyTorch -> Jittor 权重转换脚本
# 使用方式示例：
# python convert_weights.py --src ./Pretrained_Models/alexnet_caffe.pth.tar \
#     --dst ./Pretrained_Models/alexnet_caffe_jittor.pkl
# ===============================


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch AlexNet Caffe weights to Jittor format")
    parser.add_argument("--src", type=str, required=True, help="PyTorch 权重文件路径 (.pth.tar)")
    parser.add_argument("--dst", type=str, required=True, help="输出 Jittor 权重文件路径 (.pkl)")
    return parser.parse_args()


def _strip_prefix(state_dict, prefix="module."):
    new_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state[k[len(prefix) :]] = v
        else:
            new_state[k] = v
    return new_state


def main():
    args = parse_args()

    if not os.path.isfile(args.src):
        raise FileNotFoundError(f"PyTorch 权重不存在: {args.src}")

    # 1) 加载 PyTorch 权重
    state = torch.load(args.src, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        torch_state = state["state_dict"]
    else:
        torch_state = state

    # 去掉 DDP 的 module. 前缀
    torch_state = _strip_prefix(torch_state, prefix="module.")

    # 2) 构建 Jittor 模型并获取其 state_dict keys
    jt_model = models.AlexNetCaffe()
    jt_keys = jt_model.state_dict().keys()

    # 3) 映射权重（按同名 key 映射）
    converted = {}
    for k in jt_keys:
        if k in torch_state:
            converted[k] = jt.array(torch_state[k].cpu().numpy())
        else:
            print(f"[Warning] PyTorch 权重缺失 key: {k}，将保持默认初始化")

    # 4) 保存 Jittor 权重
    jt.save(converted, args.dst)
    print(f"转换完成，已保存到: {args.dst}")


if __name__ == "__main__":
    main()
