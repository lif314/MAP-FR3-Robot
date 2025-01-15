import torch
import os


base_Path = "checkpoints"
# LGDM input_channels=3
# model_name = "model_lgd_grasp_anything++"

# GenerativeResnet input_channels=3
# model_name = "det-seg-refine"

# LGGCNN input_channels=3
# model_name = "lggcnn"

#  GenerativeResnet input_channels=3
# model_name = "lgm-diff"

# GenerativeResnet input_channels=3
model_name = "lgrconvnet"

# 加载 model.state_dict() 格式
model_path = os.path.join(base_Path, model_name)  # 你的模型路径
net = torch.load(model_path)

# 保存为 Pth 格式，只保存权重（state_dict）
torch.save(net.state_dict(), os.path.join(base_Path, model_name + ".pth"))
