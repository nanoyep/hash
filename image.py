import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse

# 加载预训练模型（如果已有本地 .pth，可以先用 torchvision 加载，再保存为 state_dict 格式）
def load_model(model_path=None, device='cpu'):
    if model_path:
        # 如果提供了本地 .pth，则先创建模型再加载
        model = models.mobilenet_v3_small(pretrained=False)
        state_dict = torch.load(model_path, map_location='cpu')
        # 如果 state_dict 包含 'state_dict' 键，则提取
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model = models.mobilenet_v3_small(pretrained=True)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, input_size=224):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def extract_features(model, img_tensor, device='cpu'):
    """提取特征向量（全局平均池化后的特征）"""
    with torch.no_grad():
        # torchvision 模型的特征提取层是 features 模块
        features = model.features(img_tensor.to(device))
        # 全局平均池化
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = pooled.view(pooled.size(0), -1)
    return features.cpu().numpy().flatten()

def cosine_similarity(feat1, feat2):
    dot = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', required=True, help='Path to first image')
    parser.add_argument('--image2', required=True, help='Path to second image')
    parser.add_argument('--model_path', default=None, help='Path to .pth file (optional)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = load_model(args.model_path, args.device)

    img1 = preprocess_image(args.image1)
    img2 = preprocess_image(args.image2)

    feat1 = extract_features(model, img1, args.device)
    feat2 = extract_features(model, img2, args.device)

    sim = cosine_similarity(feat1, feat2)
    print(f"Cosine similarity: {sim:.4f}")

if __name__ == '__main__':
    main()