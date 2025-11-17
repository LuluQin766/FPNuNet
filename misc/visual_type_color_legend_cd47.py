import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 路径配置
COLOR_DICT_PATH = "/root/SAM2PATH-main/dataset_process/type_info_cd47_nuclei.json"
SAVE_PATH = "/root/SAM2PATH-main/dataset_process/type_color_legend.png"  # 可修改为任意输出路径

# 加载颜色信息
with open(COLOR_DICT_PATH, 'r') as f:
    type_info = json.load(f)

# 排序确保编号顺序展示
sorted_items = sorted(type_info.items(), key=lambda x: int(x[0]))

# 准备画布
n = len(sorted_items)
fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * n)))
ax.axis("off")

# 绘制颜色块和文字标签
for i, (type_id, (name, rgb)) in enumerate(sorted_items):
    color = tuple([v / 255.0 for v in rgb])
    rect = mpatches.Rectangle((0, i), 1, 1, facecolor=color)
    ax.add_patch(rect)
    ax.text(1.2, i + 0.5, f"ID {type_id}: {name}", va='center', fontsize=12)

ax.set_xlim(0, 4)
ax.set_ylim(0, n)
ax.invert_yaxis()

# 保存或展示
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150)
plt.close()
print(f"[✓] Type color legend saved to {SAVE_PATH}")
