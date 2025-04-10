import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  

model.classes = [0]

stream_url = "https://hls.ibb.gov.tr/ls/cam_turistik/b_eyupsultan.stream/chunklist.m3u8"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Akış açılamadı!")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()  
        cv2.imshow("YOLOv8 İnsan Tespiti", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
