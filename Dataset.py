# Dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(512, 512), num_classes=5):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size  # 统一尺寸
        self.num_classes = num_classes

        # 获取文件名集合（去掉后缀），只保留同时存在 image 和 mask 的样本
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.png'))]

        img_bases = set(os.path.splitext(f)[0] for f in img_files)
        mask_bases = set(os.path.splitext(f)[0] for f in mask_files)

        valid_bases = img_bases & mask_bases  # 只保留同时存在的

        # 按数字排序（假设文件名是数字编号）
        try:
            self.names = sorted(valid_bases, key=lambda x: int(x))
        except ValueError:
            # 如果文件名不是纯数字，按字典序排序
            self.names = sorted(valid_bases)

        if len(self.names) == 0:
            raise RuntimeError("没有找到有效的 image-mask 对！请检查路径或文件名。")

        print(f"Total valid image-mask pairs: {len(self.names)}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        base_name = self.names[idx]

        # 自动匹配 image 和 mask 的文件
        img_path = None
        mask_path = None

        for ext in ['.jpg', '.png']:
            temp = os.path.join(self.img_dir, base_name + ext)
            if os.path.exists(temp):
                img_path = temp
                break

        for ext in ['.jpg', '.png']:
            temp = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(temp):
                mask_path = temp
                break

        if img_path is None or mask_path is None:
            raise FileNotFoundError(f"未找到对应 image 或 mask: {base_name}")

        # 读取图片
        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path)

        # resize
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        # 转 tensor
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)  # [1,H,W]
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H,W]

        # ===== 自动修正 mask 超出范围的值 =====
        mask_clamped = mask.clamp(0, self.num_classes - 1)
        if not torch.equal(mask, mask_clamped):
            print(f"[Warning] {base_name}: mask 中存在超出范围的值，已修正")
            mask = mask_clamped

        # debug 输出最小最大值
        # print(f"{base_name}: mask min={mask.min().item()}, max={mask.max().item()}")

        return img, mask
