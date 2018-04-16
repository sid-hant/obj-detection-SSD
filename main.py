# importing libraries
#ssl to download imageio plugins
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
imageio.plugins.ffmpeg.download()

#detects the object
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    f = torch.from_numpy(frame_t).permute(2, 0, 1)
    f = Variable(f.unsqueeze(0))
    y = net(f)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

#imports the ssd NN
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

#takes the video and breaks into pictures
reader = imageio.get_reader('ENTER-VIDEO-TITLE-HERE.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)

#goes through the video and puts boxes around horses
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
writer.close()
print("COMPLETED")
