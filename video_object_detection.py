import cv2
import numpy as np
import time
import os

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = list()
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames() # 레이어들의 이름을 얻음
output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # 연결되지 않은 레이어의 index
# 화면에 출력 레이어를 사용하여 물체 감지 가능

action_list = ['book', 'laptop', 'phone', 'water', 'wash']
#action_list = ['water', 'wash']
# action_list = ['doorin', 'doorout'] # doorin과 doorout의 경우 자르는 축을 더 늘리던가 아니면 다른 방법 생각
# action_list = ['wash']
# door의 경우는 그대로 이미지를 사용해도 괜찮음
video_path = './save_video/'
#  real time object detection
# cap = cv2.VideoCapture(0)
for action in action_list:
    print(action + ' start...')
    # os file list in directory
    file_list = os.listdir(video_path + action)

    frame_id = 0
    if action == 'laptop' or action == 'book': # laptop, book의 경우 사람과 떨어져서도 사용 가능
                           # laptop은 coco.names에서 63번째, book은 73번째
        for fl in file_list:
            cap = cv2.VideoCapture(video_path + action + "/" + fl)

            font = cv2.FONT_HERSHEY_PLAIN
            xl, yl, wl, hl = 0, 0, 0, 0
            x, y, w, h = 0, 0, 0, 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
                    frame_id += 1 # 새 frame을 얻을때마다 +1함

                    height, width, channels = frame.shape

                    # Detecting objects
                    # cpu가 처리해야하는 크기를 줄이기 위해 416 -> 320 변경 # 작을수록 빠르지만 정확도는 떨어짐
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=True)

                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    # Showing informations on the screen
                    confidences = list()
                    class_ids = list()
                    boxes = list()
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores) # 어떤 물체로 탐지하였는지 확인하기 위해
                            if class_id == 0 or class_id == 63:
                                confidence = scores[class_id]
                                if confidence > 0.8: # 어느정도 확신하는지 confidence에 들어감
                                    # Object detected
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)

                                    # Rectangle coordinates
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)

                                    boxes.append([x, y, w, h])
                                    confidences.append(float(confidence))
                                    class_ids.append(class_id)
                            elif class_id == 73:
                                confidence = scores[class_id]
                                if confidence > 0.78: # 어느정도 확신하는지 confidence에 들어감
                                    # Object detected
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)

                                    # Rectangle coordinates
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)

                                    boxes.append([x, y, w, h])
                                    confidences.append(float(confidence))
                                    class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 0.5는 계값, 0.4는 표준값
                    for i in range(len(boxes)):
                        if i in indexes:
                            label = str(classes[class_ids[i]])
                            if label == 'laptop' or label == 'book':
                                xl, yl, wl, hl = boxes[i]
                            elif label == 'person':
                                x, y, w, h = boxes[i]
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            if frame_id % 10 == 0:
                                if x < 0: x = 0
                                if y < 0: y = 0
                                len_y = y+h; len_x = x+w
                                if xl != 0 and yl != 0:
                                    if y < yl and y+h < len_y+hl:
                                        len_y = len_y+hl
                                    if x < xl and x+w < len_x+wl:
                                        len_x = len_x+wl

                                # width와 height를 넘어버릴 경우
                                if len_x > width: len_x = width
                                if len_y > height: len_y = height

                                person_frame = frame[y:len_y, x:len_x]
                                try:
                                    current_time = time.time()
                                    # 전체 이미지에서 사람만을 특정하여 사진 저장
                                    img_person_item = './save_person_img/' + action + '/' + str(current_time) + '_' + str(int(frame_id/10)) + ".jpg"

                                    cv2.imwrite(img_person_item, person_frame)
                                    # 전체 이미지를 그대로 저장
                                    # img_total_item = './save_total_img/200717_after/' + action + '/' + str(current_time) + '_' + str(int(frame_id/10)) + ".jpg"
                                    # cv2.imwrite(img_total_item, frame)
                                except:
                                    continue

                        # cv2.imshow("Image", frame)
                        key = cv2.waitKey(1) # 0이 아니라 1밀리초만큼 기다렸다가 다음을 진행 (real time 영상이기에 1로 설정)
                        if key == 27:
                            break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
    elif action == 'doorin' or action == 'doorout':
        for fl in file_list:
            cap = cv2.VideoCapture(video_path + action + "/" + fl)

            font = cv2.FONT_HERSHEY_PLAIN
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

                    height, width, channels = frame.shape

                    # Detecting objects
                    # cpu가 처리해야하는 크기를 줄이기 위해 416 -> 320 변경 # 작을수록 빠르지만 정확도는 떨어짐
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=True)

                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    # Showing informations on the screen
                    confidences = list()
                    boxes = list()
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores) # 어떤 물체로 탐지하였는지 확인하기 위해
                            if class_id == 0: # 사람만을 분류
                                confidence = scores[class_id]
                                if confidence > 0.9: # 어느정도 확신하는지 confidence에 들어감
                                    # Object detected
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)

                                    # Rectangle coordinates
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)

                                    boxes.append([x, y, w, h])
                                    confidences.append(float(confidence))

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 0.5는 계값, 0.4는 표준값
                    for i in range(len(boxes)):
                        if i in indexes:
                            frame_id += 1 # 새 frame을 얻을때마다 +1함
                            x, y, w, h = boxes[i]
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            if frame_id % 5 == 0:
                                if x < 0: x = 0
                                if y < 0: y = 0
                                person_frame = frame[:, x-int(w/2):x+w+int(w/2)]
                                try:
                                    current_time = time.time()
                                    # 전체 이미지에서 사람만을 특정하여 사진 저장
                                    img_person_item = './save_person_img/200717_after/' + action + '/' + str(current_time) + '_' + str(int(frame_id/10)) + ".jpg"
                                    cv2.imwrite(img_person_item, person_frame)
                                    # 전체 이미지를 그대로 저장
                                    # img_total_item = './save_total_img/200705_after/' + action + '/' + str(current_time) + '_' + str(int(frame_id/10)) + ".jpg"
                                    # cv2.imwrite(img_total_item, frame)
                                except:
                                    continue

                    # cv2.imshow("Image", frame)
                    key = cv2.waitKey(1) # 0이 아니라 1밀리초만큼 기다렸다가 다음을 진행 (real time 영상이기에 1로 설정)
                    if key == 27:
                        break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
    else:
        for fl in file_list:
            cap = cv2.VideoCapture(video_path + action + "/" + fl)

            font = cv2.FONT_HERSHEY_PLAIN
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is True:
                    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
                    frame_id += 1 # 새 frame을 얻을때마다 +1함

                    height, width, channels = frame.shape

                    # Detecting objects
                    # cpu가 처리해야하는 크기를 줄이기 위해 416 -> 320 변경 # 작을수록 빠르지만 정확도는 떨어짐
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=True)

                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    # Showing informations on the screen
                    confidences = list()
                    boxes = list()
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores) # 어떤 물체로 탐지하였는지 확인하기 위해
                            if class_id == 0: # 사람만을 분류
                                confidence = scores[class_id]
                                if confidence > 0.95: # 어느정도 확신하는지 confidence에 들어감
                                    # Object detected
                                    center_x = int(detection[0] * width)
                                    center_y = int(detection[1] * height)
                                    w = int(detection[2] * width)
                                    h = int(detection[3] * height)

                                    # Rectangle coordinates
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)

                                    boxes.append([x, y, w, h])
                                    confidences.append(float(confidence))

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 0.5는 계값, 0.4는 표준값
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            if frame_id % 10 == 0:
                                if x < 0: x = 0
                                if y < 0: y = 0
                                person_frame = frame[y:y+h, x:x+w]
                                try:
                                    current_time = time.time()
                                    # 전체 이미지에서 사람만을 특정하여 사진 저장
                                    img_person_item = './save_person_img/' + action + '/' + str(current_time) + '_' + str(int(frame_id/10)) + ".jpg"
                                    cv2.imwrite(img_person_item, person_frame)
                                    # 전체 이미지를 그대로 저장
                                    # img_total_item = './save_total_img/200705_after/' + action + '/' + str(current_time) + '_' + str(int(frame_id/10)) + ".jpg"
                                    # cv2.imwrite(img_total_item, frame)
                                except:
                                    continue

                    # cv2.imshow("Image", frame)
                    key = cv2.waitKey(1) # 0이 아니라 1밀리초만큼 기다렸다가 다음을 진행 (real time 영상이기에 1로 설정)
                    if key == 27:
                        break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
