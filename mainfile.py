import os

from ultralytics import YOLO
import cv2
import imutils

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'test1.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

def check_collision(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)


model = YOLO("C:\\Users\\AYUSH SHUKLA\\Desktop\\v8\\ball\\runs\\detect\\train27\\weights\\last.pt") 

threshold = 0.2

while ret:
    frame = imutils.resize(frame, width=640)
    results = model(frame)[0]
    class_0_boxes = []
    class_1_boxes = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            class_id = int(class_id)
            if class_id == 0:
                class_0_boxes.append((x1, y1, x2, y2))
            elif class_id == 1:
                class_1_boxes.append((x1, y1, x2, y2))
            if(results.names[class_id].upper()=="BALLOON"):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[class_id].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(frame, results.names[class_id].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3, cv2.LINE_AA)

    for box_0 in class_0_boxes:
        for box_1 in class_1_boxes:
            if check_collision(box_0, box_1):
                print("Collision detected between class 0 and class 1!")
    cv2.imshow("frame", frame)
    cv2.waitKey(10)
    ret, frame = cap.read()
    

cap.release()
out.release()
cv2.destroyAllWindows()