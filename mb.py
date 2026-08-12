#!/usr/bin/env python3
"""
MobileNetV4 图像相似度比较工具
适用于 Termux / Linux / macOS

用法:
    python img_similarity.py image1.jpg image2.jpg
    python img_similarity.py image1.jpg image2.jpg --model mobilenetv4_hybrid_small
    python img_similarity.py image1.jpg image2.jpg --model mobilenetv4_conv_medium --no-pretrained
"""
import os
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"
import argparse
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# ---------- 可用的 MobileNetV4 模型列表 ----------
MOBILENETV4_MODELS = [
    "mobilenetv4_conv_small_035",
    "mobilenetv4_conv_small_050",
    "mobilenetv4_conv_small",      # ✅ 默认，最快最小
    "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large",
    "mobilenetv4_conv_aa_medium",
    "mobilenetv4_conv_aa_large",
    "mobilenetv4_hybrid_medium_075",
    "mobilenetv4_hybrid_medium",
    "mobilenetv4_hybrid_large",
    "mobilenet_hybrid_large_075",
]

# 你也可以使用其他模型（只要是 timm 支持的即可）
OTHER_SUGGESTED_MODELS = [
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "resnet18",
    "vit_small_patch16_224",
]


def list_available_models():
    """列出所有推荐的模型"""
    print("\n========== MobileNetV4 系列 ==========")
    for m in MOBILENETV4_MODELS:
        print(f"  • {m}")
    print("\n========== 其他轻量推荐模型 ==========")
    for m in OTHER_SUGGESTED_MODELS:
        print(f"  • {m}")
    print()


def load_model(model_name: str, pretrained: bool = True):
    """
    加载模型并返回 (model, transform, feature_dim)
    模型被改造为特征提取器（去掉分类头）
    """
    print(f"[*] 加载模型: {model_name} (pretrained={pretrained})")

    # 创建模型
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    # num_classes=0 → 返回池化后的特征向量（去掉分类头）

    model.eval()

    # 获取模型对应的预处理配置
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # 获取特征维度
    # 用假数据跑一次获取维度
    with torch.no_grad():
        dummy = torch.randn(1, 3, config["input_size"][0], config["input_size"][1])
        feat = model(dummy)
        feature_dim = feat.shape[1]

    print(f"[*] 输入尺寸: {config['input_size']}, 特征维度: {feature_dim}")
    return model, transform, feature_dim


def extract_features(model, transform, image_path: str, device: torch.device):
    """
    对单张图片提取特征向量
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[!] 无法读取图片 {image_path}: {e}")
        sys.exit(1)

    # 预处理
    tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        features = model(tensor)  # shape: (1, feature_dim)

    # L2 归一化（让余弦相似度计算更稳定）
    features = nn.functional.normalize(features, p=2, dim=1)

    return features.cpu().numpy().flatten()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """余弦相似度，返回 [0, 1]，越大越相似"""
    # 已做 L2 归一化，直接点积即可
    sim = np.dot(vec1, vec2)
    # 限制在 [0, 1]
    sim = max(0.0, min(1.0, float(sim)))
    return sim


def main():
    parser = argparse.ArgumentParser(
        description="使用 MobileNetV4 比较两张图片的相似度"
    )
    parser.add_argument("image1", help="第一张图片路径")
    parser.add_argument("image2", help="第二张图片路径")
    parser.add_argument(
        "--model", "-m",
        default="mobilenetv4_conv_small",
        help="模型名称（默认: mobilenetv4_conv_small）"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不使用预训练权重（不推荐）"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出推荐模型并退出"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="推理设备（Termux 下只能用 cpu）"
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 加载模型
    model, transform, feat_dim = load_model(
        args.model,
        pretrained=not args.no_pretrained
    )
    model = model.to(device)

    # 提取特征
    print(f"[*] 提取特征: {args.image1}")
    vec1 = extract_features(model, transform, args.image1, device)

    print(f"[*] 提取特征: {args.image2}")
    vec2 = extract_features(model, transform, args.image2, device)

    # 计算相似度
    similarity = cosine_similarity(vec1, vec2)

    # 输出结果
    print("\n" + "=" * 50)
    print(f"  模型: {args.model}")
    print(f"  图片1: {args.image1}")
    print(f"  图片2: {args.image2}")
    print(f"  特征维度: {feat_dim}")
    print(f"  -----------------------------------")
    print(f"  🔥 相似度: {similarity:.4f}  ({similarity * 100:.2f}%)")
    print("=" * 50)

    # 给出直观判断
    if similarity > 0.95:
        verdict = "几乎相同 / 重复图片"
    elif similarity > 0.85:
        verdict = "高度相似"
    elif similarity > 0.70:
        verdict = "较为相似"
    elif similarity > 0.50:
        verdict = "有一定相似性"
    else:
        verdict = "不相似"
    print(f"  📋 判断: {verdict}\n")


if __name__ == "__main__":
    main()

