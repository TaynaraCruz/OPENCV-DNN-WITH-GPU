import numpy as np
import cv2
import time

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open('coco.names', 'r') as f:
  class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

#USE GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

VideoCap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

while(VideoCap.isOpened()):
  ret, frame = VideoCap.read()
  detected = False
  if not ret:
    break
  
  new_frame_time = time.time()
  fps = 1/(new_frame_time-prev_frame_time)
  prev_frame_time = new_frame_time

  fps = int(fps)
  fps = str(fps)

  classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

  for (classid, score, box) in zip(classes, scores, boxes):
    color = COLORS[int(classid) % len(COLORS)]
    label = f'{class_names[classid]} : {score:.2f}'
    cv2.rectangle(frame, box, color, 2)
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f'FPS: {fps}', (7, 70), 1, 3, (100, 255, 0), 3, cv2.LINE_AA)
  
  cv2.imshow('Frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    VideoCap.release()
    cv2.destroyAllWindows()
    break