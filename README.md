# object-detection-with-YOLOv3
- 사람의 행동 인식을 위해 사람을 탐지 그 주변을 잘라내는 데이터 전처리 코드

## 전처리전 필요한 파일 다운로드 (https://pjreddie.com/darknet/yolo/)
- yolov3.weights or yolov3-tiny.weights
- yolov3.cfg or yolov3-tiny.cfg
- coco.names

## video_object_detection.py
- 지정된 사람의 행동 (책 읽는 행동 - book, 노트북 하는 행동 - laptop, 핸드폰 하는 행동 - phone, 물 마시는 행동 - water, 설거지하는 행동 - wash)을 인식할 수 있게 이미지를 사람 + 행동에 맞는 물체에 맞게 잘라내어 외부 노이즈를 제거
- 노이즈가 제거된 이미지는 지정된 행동 폴더에 저장

## 기타
- 해당 repository 생성 날짜 : 2020-12-28 / 1:47
- 2021년 새해 복 많이 받으세요~
