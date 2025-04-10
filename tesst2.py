import cv2
from ultralytics import YOLO
import numpy as np

# Video dosyası
video_path = "peoplecount1.mp4"
cap = cv2.VideoCapture(video_path)

# YOLOv8 modelini yükle
model = YOLO('yolov8s.pt')

# Sayaçlar
entry_count = 0
exit_count = 0
total_people = 0

# Kapı alanı (dikey bölge)
door_area = {
    'left': 1350,     # Sol sınır
    'right': 1500,    # Sağ sınır
    'top': 540,   # Üst sınır
    'bottom': 2100,  # Alt sınır
    'color': (255, 255, 0)  # Sarı renk (BGR)
}

# Kişilerin geçiş durumlarını takip etmek için
person_tracks = {}  # track_id -> {"last_x": x, "last_y": y, "counted": False, "in_area": False}

def check_area_crossing(current_x, current_y, last_x, last_y):
    """Alan geçişini ve yönünü kontrol et"""
    # Kişinin merkez noktası alanın içinde mi?
    in_area = (door_area['left'] <= current_x <= door_area['right'] and 
               door_area['top'] <= current_y <= door_area['bottom'])
    
    # Son konum alanın içinde miydi?
    was_in_area = (door_area['left'] <= last_x <= door_area['right'] and 
                   door_area['top'] <= last_y <= door_area['bottom'])
    
    # Alan geçişi kontrolü
    if not was_in_area and in_area:
        # Alanın sol tarafından mı, sağ tarafından mı girdi?
        if last_x < door_area['left']:
            return "entry", True
        elif last_x > door_area['right']:
            return "exit", True
    elif was_in_area and not in_area:
        # Alanın sol tarafına mı, sağ tarafına mı çıktı?
        if current_x < door_area['left']:
            return "exit", False
        elif current_x > door_area['right']:
            return "entry", False
    
    return None, in_area

def draw_door_area(frame):
    # Kapı alanını çiz
    cv2.rectangle(frame, 
                 (door_area['left'], door_area['top']), 
                 (door_area['right'], door_area['bottom']), 
                 door_area['color'], 2)
    
    # Orta çizgiyi çiz (referans için)
    mid_x = (door_area['left'] + door_area['right']) // 2
    cv2.line(frame, (mid_x, door_area['top']), (mid_x, door_area['bottom']), (0, 0, 255), 1)
    
    # Sayaç bilgilerini göster
    cv2.putText(frame, f"Giris: {entry_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cikis: {exit_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Icerdeki Kisi: {total_people}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Video işleme döngüsü
for result in model.track(source=video_path, show=False, stream=True, conf=0.3, iou=0.3, classes=0):
    frame = result.orig_img
    
    if result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()  # x, y, width, height
        track_ids = result.boxes.id.int().cpu().tolist()  # tracking IDs
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            cx = int(x)  # merkez x koordinatı
            cy = int(y)  # merkez y koordinatı
            
            # Yeni tespit edilen kişi
            if track_id not in person_tracks:
                person_tracks[track_id] = {
                    "last_x": cx,
                    "last_y": cy,
                    "counted": False,
                    "in_area": False
                }
            
            # Alan geçişini kontrol et
            crossing, in_area = check_area_crossing(cx, cy, 
                                                  person_tracks[track_id]["last_x"], 
                                                  person_tracks[track_id]["last_y"])
            
            # Kişinin alan içinde olma durumunu güncelle
            person_tracks[track_id]["in_area"] = in_area
            
            # Geçiş varsa sayım yap
            if crossing and not person_tracks[track_id]["counted"]:
                if crossing == "entry":
                    entry_count += 1
                    total_people += 1
                else:  # exit
                    exit_count += 1
                    total_people = max(0, total_people - 1)
                person_tracks[track_id]["counted"] = True
            
            # Kişinin rengini belirle
            if person_tracks[track_id]["in_area"]:
                color = (0, 255, 255)  # Sarı - alan içinde
            elif person_tracks[track_id]["counted"]:
                color = (0, 255, 0) if cx > (door_area['left'] + door_area['right'])/2 else (0, 0, 255)
            else:
                color = (255, 255, 255)  # Beyaz - alan dışında
            
            # Çizgiden yeterince uzaklaşınca sayım durumunu sıfırla
            if abs(cx - (door_area['left'] + door_area['right']) / 2) > 200:
                person_tracks[track_id]["counted"] = False
            
            # Son konumu güncelle
            person_tracks[track_id]["last_x"] = cx
            person_tracks[track_id]["last_y"] = cy
            
            # Kişiyi çerçeve içine al
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    draw_door_area(frame)
    cv2.imshow("Market Giris-Cikis Sayaci", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
