import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8s.pt')  
model.classes = [0]  # Sadece insan sınıfı

stream_url = "https://hls.ibb.gov.tr/ls/cam_turistik/b_kapalicarsi.stream/chunklist.m3u8"
cap = cv2.VideoCapture(stream_url)

# Sayaç
counter = 0

# Takip için merkezler ve geçmiş
tracked_objects = {}  # object_id: (x, y)
object_id = 0

# Nesnenin geçip geçmediğini takip etmek için
counted_ids = set()

if not cap.isOpened():
    print("Akış açılamadı!")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        frame_height, frame_width = annotated_frame.shape[:2]

        # Çizgi: Alttan %30 yukarıda
        line_position = int(frame_height * 0.7)

        # Çizgiyi çiz
        cv2.line(annotated_frame, (0, line_position), (frame_width, line_position), (0, 255, 255), 2)

        # Şu anki merkezler
        new_centroids = []

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            new_centroids.append((cx, cy))

            # Merkezleri çiz
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

        # Mevcut merkezleri eski merkezlerle eşleştir
        updated_objects = {}
        for centroid in new_centroids:
            # En yakın eski nesne bulun
            min_dist = float('inf')
            min_id = None
            for obj_id, prev_centroid in tracked_objects.items():
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if dist < min_dist and dist < 50:  # Eğer çok uzakta değilse (eşik: 50 px)
                    min_dist = dist
                    min_id = obj_id

            if min_id is not None:
                updated_objects[min_id] = centroid
            else:
                # Yeni bir nesne
                updated_objects[object_id] = centroid
                object_id += 1

        tracked_objects = updated_objects

        # Çizgiyi geçenleri kontrol et
        for obj_id, (cx, cy) in tracked_objects.items():
            if obj_id in counted_ids:
                continue  # Zaten sayıldıysa atla

            prev_cy = cy - 10  # Önceki y tahmini (küçük bir fark veriyoruz)

            if prev_cy < line_position <= cy:
                counter += 1
                counted_ids.add(obj_id)

        # Sayacı ekrana yaz
        cv2.putText(annotated_frame, f"Yukaridan Asagi Gecenler: {counter}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Insan Tespiti", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
