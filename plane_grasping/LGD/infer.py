import torch


# LGDM input_channels=3
# model_path = "checkpoints/model_lgd_grasp_anything++"

# GenerativeResnet input_channels=3
# model_path = "checkpoints/det-seg-refine"

# LGGCNN input_channels=3
# model_path = "checkpoints/lggcnn"

#  GenerativeResnet input_channels=3
# model_path = "checkpoints/lgm-diff"

# GenerativeResnet input_channels=3
model_path = "checkpoints/lgrconvnet"

net =  torch.load(model_path)

print("Net: ", net)