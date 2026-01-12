import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # 导入tqdm库
from Unet import Unet
from Dataset import SegDataset


# =====================
# SAVE
# =====================
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# =====================
# Device
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# Hyper params
# =====================
num_classes = 5
epochs = 2
batch_size = 4
lr = 1e-3
save_best = True


def main():
    best_loss = float("inf")
    # =====================
    # Dataset
    # =====================
    dataset = SegDataset(
        img_dir="dataset/raw",
        mask_dir="dataset/masks"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=(device.type == "cuda")
    )

    # =====================
    # Model / Optimizer / Loss
    # =====================
    model = Unet(in_ch=1, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # =====================
    # Training Loop
    # =====================
    # 外层epoch进度条
    for epoch in tqdm(range(epochs), desc="Total Training Progress", unit="epoch"):
        model.train()
        total_loss = 0.0

        # 内层batch进度条
        pbar = tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for step, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(device)  # [B, 1, H, W]
            masks = masks.to(device).long()

            preds = model(imgs)  # [B, C, H, W]
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条的实时显示信息
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            # ===== sanity check（只在第一轮打印一次）=====
            if epoch == 0 and step == 0:
                print("\nmask unique:", torch.unique(masks))
                print("pred unique:", torch.unique(preds.argmax(1)))

        pbar.close()  # 关闭当前epoch的batch进度条
        avg_loss = total_loss / len(loader)
        # 在epoch进度条中打印本轮平均loss
        tqdm.write(f"Epoch [{epoch + 1:03d}/{epochs}]  Avg Loss: {avg_loss:.4f}")

        # =====================
        # Save model (last)
        # =====================


        save_path = os.path.join(save_dir, f"unet_epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, save_path)

        # =====================
        # Save best model
        # =====================
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "unet_best.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, best_path)
            tqdm.write(f"✔ Best model saved (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()