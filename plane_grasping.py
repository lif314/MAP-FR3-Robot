import yaml
import argparse
import torch
import os
from PIL import Image
import numpy as np
from plane_grasping.processing.camera import CameraData
import matplotlib.pyplot as plt
from plane_grasping.visualisation.plot import plot_results

from config import load_config
from skimage.filters import gaussian

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for running Plane-Grasping.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='Path to logs')
    parser.add_argument('--online', type=bool, default=False,
                        help='Online plane grasping with real robot')
    
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    args = parser.parse_args()
    return args

def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))
        


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img


from plane_grasping.LGD.models import get_network
from plane_grasping.GGCNN.models.ggcnn import GGCNN
from plane_grasping.GGCNN.models.ggcnn2 import GGCNN2

from plane_grasping.LGD.models.lgrconvnet3 import GenerativeResnet


    

def main():
    parsed_args = parse_args()

    config = load_config(parsed_args.config, 'configs/grcnn.yaml')

    print("config: ", config)
    # logs
    os.makedirs(config["log_dir"], exist_ok=True)

    


    # input_channels = 3 * args.use_rgb + 1 * args.use_depth

    # # model_path = "models/plane_grasping/GRCNN/checkpoints/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97.pth"
    
    # # model_path = "models/plane_grasping/GGCNN/checkpoints/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
    # # model_path = "models/plane_grasping/GGCNN/checkpoints/ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt"
    # # model_path = "models/plane_grasping/LGD/checkpoints/model_lgd_grasp_anything++.pth"
    # model_path = "models/plane_grasping/LGD/checkpoints/lgrconvnet.pth"
    # # model_path = "models/plane_grasping/LGD/checkpoints/lggcnn.pth"

    # # GRCNN
    # # network = get_network(args.network)
    # # net = network(
    # #     input_channels=input_channels,
    # #     dropout=args.use_dropout,
    # #     prob=args.dropout_prob,
    # #     channel_size=args.channel_size
    # # )

    # # net = GGCNN(input_channels=input_channels)
    # # net = GGCNN2(input_channels=input_channels)
    # # net.load_state_dict(torch.load(model_path))

    # # network = get_network("lggcnn")

    # # net = network(
    # #     input_channels=3
    # # )

    # net = GenerativeResnet(input_channels=3)

    # net.load_state_dict(torch.load(model_path))

    # net = net.to("cuda")

    # rgb_path = "testdata/images/cmp2.png"
    # # rgb_path = "testdata/rgb/0000.png"
    # depth_path = "testdata/images/hmp2.png"

    # pic = Image.open(rgb_path, 'r')
    # rgb = np.array(pic)
    # pic = Image.open(depth_path, 'r')
    # depth = np.expand_dims(np.array(pic), axis=2)

    # img_data = CameraData(include_depth=False, include_rgb=True)

    # x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

    # query = ['grasp the blue box'] *1
    
    # with torch.no_grad():
    #     xc = x.to("cuda")
    #     print("xc: ", xc.shape)
    #     # pred = net.predict(xc)
    #     pos_output, cos_output, sin_output, width_output = net(xc, query, query)

    #     # q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
    #     q_img, ang_img, width_img = post_process_output(pos_output, cos_output, sin_output, width_output)

    #     fig = plt.figure(figsize=(10, 10))
    #     plot_results(fig=fig,
    #                     rgb_img=img_data.get_rgb(rgb, False),
    #                     grasp_q_img=q_img,
    #                     grasp_angle_img=ang_img,
    #                     no_grasps=1,
    #                     grasp_width_img=width_img)
    #     fig.savefig('img_result.png')


if __name__ == '__main__':
    main()