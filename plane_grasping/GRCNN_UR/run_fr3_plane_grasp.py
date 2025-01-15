from plane_grasp_fr3 import PlaneGraspFR3

import time
if __name__ == '__main__':
    g = PlaneGraspFR3(
        saved_model_path='./trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
        visualize=True,
        include_rgb=True
    )
    if g.panda.move_to_start():
        if g.gripper.homing():
            grasp_result =[]
            # iter=0
            while True:
                grasp_success = g.generate()
                if grasp_success:
                    grasp_result.append(True)
                else:
                    grasp_result.append(False)
                # end
                # if (iter>=2) and (not grasp_result[iter]) and (not grasp_result[iter-1]) and (not grasp_result[iter-2]):
                #     print('grasp_result_array:',grasp_result)
                #     break
                # iter += 1
                time.sleep(0.2)
    
