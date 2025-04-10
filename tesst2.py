import cv2
from ultralytics import YOLO
from TrackingModule import EuclideanDistTracker

# Kamera veya yayın linki
stream_url = "https://hls.ibb.gov.tr/ls/cam_turistik/b_kapalicarsi.stream/chunklist.m3u8"
cap = cv2.VideoCapture(stream_url)

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # Daha istersen yolov8s.pt kullanabilirsin

# Takip için tracker
tracker = EuclideanDistTracker()

# Sayaç
counter = 0

# Çizgi koordinatları (örnek: kapının önü gibi)
line_position = 370
line_color = (0, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 ile tahmin yap
    results = model(frame, stream=True)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.4:  # class 0: 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h

    # Takibi yap
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cx = x + w // 2
        cy = y + h // 2

        # Nesne çizgiyi geçerse
        if cy < line_position + 10 and cy > line_position - 10:
            counter += 1
            print(f"Kişi geçti! Toplam: {counter}")

        # Ekrana çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Çizgi
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), line_color, 2)
    cv2.putText(frame, f"Sayac: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)

    cv2.imshow("YOLOv8 - Gecis Sayaci", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
