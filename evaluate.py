import os

import torch
from soupsieve.util import lower

from Unet import Unet

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

TEST_DIR = "./dataset/tests"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5

model = Unet(in_ch = 1,num_classes = num_classes).to(device)

#加载权重
ckpt = torch.load("./checkpoints/unet_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

#加载图片文件
img_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith('jpg')]

for img_file in img_files:
    img = Image.open(os.path.join(TEST_DIR, img_file)).convert('L')
    img = img.resize((512,512))

    #转为tensor [1,1,H,W]
    img_tensor = torch.from_numpy(np.array(img,dtype=np.float32) / 255.0)   #归一化
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)            #[B,1,H,W]

    with torch.no_grad():
        pred = model(img_tensor)                        #[1,C,H,W]
        pred_mask = torch.argmax(pred, dim=1)           #[1,H,W]
        pred_mask = pred_mask.squeeze(0).cpu().numpy()  #[H,W]

    #可视化呈现
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1,2,2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="tab10")
    plt.show()
