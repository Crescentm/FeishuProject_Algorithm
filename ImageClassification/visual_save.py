# 最终结果可视化与保存，csv文件生成
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import efficientnet_b7
from dataset import build_data_set
from torch.utils.data import DataLoader
import os
import pandas as pd
import tqdm

# ------------ 参数设定 ------------
dest_image_size = (256, 256)
label_map = {0: "cat", 1: "dog", 2: "other"}

# ------------ Grad-CAM容器 ------------
features_deep, grads_deep = None, None
features_shallow, grads_shallow = None, None

def hook_deep_fwd(m, i, o):
    global features_deep
    features_deep = o.detach()

def hook_deep_bwd(m, gi, go):
    global grads_deep
    grads_deep = go[0].detach()

def hook_shallow_fwd(m, i, o):
    global features_shallow
    features_shallow = o.detach()

def hook_shallow_bwd(m, gi, go):
    global grads_shallow
    grads_shallow = go[0].detach()

# ------------ 模型加载 ------------
model = efficientnet_b7(num_classes=3)
model.load_state_dict(torch.load("/home/jh/Blog/code/ckpts_eff/best_model.pth", map_location="cpu"))
model.eval()

# 注册钩子：浅层和深层
layer_deep = model.features.top[0]     # 深层
layer_shallow = model.features[6]      # 浅层
layer_deep.register_forward_hook(hook_deep_fwd)
layer_deep.register_full_backward_hook(hook_deep_bwd)
layer_shallow.register_forward_hook(hook_shallow_fwd)
layer_shallow.register_full_backward_hook(hook_shallow_bwd)

# ------------ 数据加载 ------------
valid_dataset = build_data_set(dest_image_size, "/home/jh/Blog/dataset/val", is_train=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# ------------ Grad-CAM生成函数 ------------
def compute_cam(fmap, grad, input_shape):
    weights = grad.mean(dim=(1,2))  # [C]
    cam = torch.relu((weights[:, None, None] * fmap).sum(dim=0))  # [H, W]
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_shape, mode='bilinear', align_corners=False)
    return cam.squeeze().cpu().numpy()

# ------------ 创建结果文件夹 ------------
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
for label in label_map.values():
    os.makedirs(os.path.join(result_dir, label), exist_ok=True)

# ------------ 保存结果到CSV ------------
csv_data = {
    "file_path": [],
    "file_label": [],
    "pred_label": [],
    "pred_prob": []
}
# ------------ 处理验证集 ------------
for batch_idx, (images, targets) in enumerate(tqdm.tqdm(valid_loader, desc="Processing validation images")):
    # 获取文件路径和标签
    file_path = valid_dataset.samples[batch_idx][0]  # 使用 batch_idx 获取样本路径
    file_label = label_map[targets.item()]
    
    # 推理
    input_tensor = images
    logits = model(input_tensor)
    pred_class = logits.argmax(dim=1).item()
    pred_label = label_map[pred_class]
    prob = torch.softmax(logits, dim=1)[0, pred_class].item()

    # 生成CAM
    model.zero_grad()
    logits[0, pred_class].backward(retain_graph=True)
    cam_deep = compute_cam(features_deep[0], grads_deep[0], input_tensor.shape[2:])
    cam_shallow = compute_cam(features_shallow[0], grads_shallow[0], input_tensor.shape[2:])
    cam_fused = 0.5 * cam_deep + 0.5 * cam_shallow
    cam_fused = (cam_fused - cam_fused.min()) / (cam_fused.max() + 1e-8)

    # 生成图像
    orig_img = ((input_tensor[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
    heatmap = plt.get_cmap("jet")(cam_fused)[..., :3]
    overlay = np.clip(0.4 * heatmap + 0.6 * orig_img, 0, 1)

    # 保存单张包含三个子图的图像
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(result_dir, pred_label)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    axes[1].imshow(cam_fused, cmap="jet")
    axes[1].axis("off")
    axes[1].set_title("Fused Grad-CAM")

    axes[2].imshow(overlay)
    axes[2].axis("off")
    axes[2].set_title(f"Overlay: {pred_label} ({prob:.2%})")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_filename}_overlay.png"), bbox_inches="tight")
    plt.close(fig)

    # 记录到CSV
    csv_data["file_path"].append(file_path)
    csv_data["file_label"].append(file_label)
    csv_data["pred_label"].append(pred_label)
    csv_data["pred_prob"].append(prob)

# 保存CSV
df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(result_dir, "result.csv"), index=False)
print("Results saved to 'result' folder and 'result.csv'.")