import os
import json

import numpy as np
from PIL import Image, ImageDraw

# 类别映射：标签名 -> 掩码值（单通道灰度图，值范围0-255）
CLASS_MAP = {
    "jingmai": 1,
    "dongmai": 2,
    "jirouzuzhi": 3,
    "shenjing": 4
}

# 路径配置（请确认路径是否正确）
IMG_DIR = "../UBPD/raw_Images"  # 原始图片目录
JSON_DIR = "../UBPD/masked_images"  # JSON标注文件目录1
MASK_DIR = "../dataset/masks"  # 生成的掩码图片保存目录
os.makedirs(MASK_DIR, exist_ok=True)

# 统计变量，方便验证
processed_count = 0
skipped_count = 0
empty_annotation_count = 0

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
json_files = sorted(json_files, key=lambda x: int(os.path.splitext(x)[0]))
# 遍历所有JSON文件
for json_name in json_files:
    # 只处理.json文件
    if not json_name.endswith(".json"):
        skipped_count += 1
        continue

    # 提取文件前缀（去掉.json）
    base_name = json_name.replace(".json", "")
    # 拼接图片和JSON的完整路径
    img_path = os.path.join(IMG_DIR,f"{base_name}.jpg")
    json_path = os.path.join(JSON_DIR,json_name)

    # 检查图片是否存在
    if not os.path.exists(img_path):
        print(f"图片不存在：{img_path}，跳过该JSON文件：{json_name}")
        skipped_count += 1
        continue

    #读取图片，获取尺寸
    try:
        img = Image.open(img_path)
        w, h = img.size
        print(f"处理文件：{json_name} | 图片尺寸：{w}x{h}")
    except Exception as e:
        print(f"读取图片失败：{img_path}，错误：{e}")
        skipped_count += 1
        continue

    # 创建单通道掩码图 （ L模式：8位灰度图，初始值0）
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # 读取JSON标注文件
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON失败：{json_path}，错误：{e}")
        skipped_count += 1
        continue

    # 检查标注是否为空
    shapes = data.get("shapes", [])
    if not shapes:
        # print(f"该JSON无标注：{json_name}，生成全0掩码")
        empty_annotation_count += 1
        # 即使无标注，也保存全0掩码
        mask.save(os.path.join(MASK_DIR, f"{base_name}.jpg"))
        processed_count += 1
        continue

    #遍历所有标注形状，绘制掩码
    annotation_count = 0
    for shape in shapes:
        label = shape.get("label", "")
        # 跳过不在CLASS_MAP中的标签

        if label not in CLASS_MAP:
            print(f"未匹配的标签：{label}，跳过该标注")
            continue
        # print(label,'\n')
        # 获取类别对应的掩码值
        class_id = CLASS_MAP[label]

        points = shape.get("points", [])
        # for i in points:
        #     print(i," ")

        # 多边形至少三个点
        if len(points) < 3:
            print(f"无效多边形：标签={label}，坐标点数量={len(points)}，跳过")
            continue

        # 坐标转成元组（如 [(x1,y1), (x2,y2), ...]）
        polygon = [tuple(coord) for coord in points]
        # print(label, polygon[:3])

        # 绘制多边形，填充对应类别值
        # 1 2 3 4在肉眼看来是全黑的，如果需要人眼辨别，将数值*50
        # 但是不能够这么做，交叉熵函数只支持target = 8
        # draw.polygon(polygon, fill=class_id * 50)

        draw.polygon(polygon, fill=class_id)
        annotation_count += 1

    # 保存掩码图片
    mask_save_path = os.path.join(MASK_DIR, f"{base_name}.png")
    mask.save(mask_save_path)
    processed_count += 1
    # print(f"保存掩码图片：{mask_save_path} | 有效标注数：{annotation_count}\n")

# debug
print("=" * 50)
print(f"处理统计：")
print(f"成功处理文件数：{processed_count}")
print(f"跳过文件数（非JSON/图片不存在等）：{skipped_count}")
print(f"无标注的文件数：{empty_annotation_count}")
print(f"掩码图片保存目录：{MASK_DIR}")