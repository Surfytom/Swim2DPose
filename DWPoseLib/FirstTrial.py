import torch
import cv2
import numpy as np

#model = torch.load("DWPoseLib/dw-ll_ucoco_384.pth")

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

#config_file = 'DWPoseLib\cspnext-l_udp_8xb64-210e_coco-wholebody-256x192.py'
#config_file = "DWPoseLib/rtmpose-l_8xb64-270e_ubody-wholebody-256x192.py"
#config_file = "DWPoseLib/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py"
config_file = "DWPoseLib/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"

checkpoint_file = 'DWPoseLib/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, 'input_image.jpeg')

keyPoints = results[0].pred_instances.keypoints[0]

keyPoints = np.array(keyPoints).astype(np.intp)

image = cv2.imread("input_image.jpeg")

shape = np.shape(image)

for x, y in keyPoints:
    cv2.circle(image, (x, y), 15, (0, 0, 255), -1)

cv2.imshow("test1", cv2.resize(image, ((shape[0] // 6), (shape[1] // 6))))

cv2.waitKey(0)
cv2.destroyAllWindows()