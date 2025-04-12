import cv2
from ultralytics import YOLO
import numpy as np

stream_url = "https://hls.ibb.gov.tr/ls/cam_turistik/b_kapalicarsi.stream/chunklist.m3u8"
cap = cv2.VideoCapture(stream_url)

model = YOLO('yolo11n.pt')

entry_count = 0
exit_count = 0
total_people = 0
area1 = [(748, 498), (741, 470), (986, 394), (1026, 430)]  # İlk alan için koordinatlar
area2 = [(748, 498), (741, 470), (986, 394), (1026, 430)]  # İkinci alan için koordinatlar
"""
 # Eğerki farklı bir alan için test yapmak istereniz bu kodu aktif edin ve üsteki alanın kordinatlarını değiştirin.
 # Ve aşağıdaki cv2.setMouseCallback kodunu aktif edin.
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)
"""
person_tracks = {} 

def check_area_crossing(current_x, current_y, last_x, last_y):
    """Alan geçişini ve yönünü kontrol et"""
    in_area1 = cv2.pointPolygonTest(np.array(area1), (current_x, current_y), False) >= 0 
    in_area2 = cv2.pointPolygonTest(np.array(area2), (current_x, current_y), False) >= 0
    
    # Son konum alanın içinde miydi?
    was_in_area1 = cv2.pointPolygonTest(np.array(area1), (last_x, last_y), False) >= 0
    was_in_area2 = cv2.pointPolygonTest(np.array(area2), (last_x, last_y), False) >= 0
    
    in_area = in_area1 or in_area2
    was_in_area = was_in_area1 or was_in_area2
    
    # Alan geçişi kontrolü
    if not was_in_area and in_area:
        # Giriş yönü tespiti (y koordinatına göre)
        if current_y < last_y:
            return "entry", True
        else:
            return "exit", True
    elif was_in_area and not in_area:
        # Çıkış yönü tespiti (y koordinatına göre)
        if current_y < last_y:
            return "exit", False
        else:
            return "entry", False
    
    return None, in_area

def draw_areas(frame):
    # Alanları çiz
    cv2.polylines(frame, [np.array(area1)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(area2)], True, (0, 255, 255), 2)
    
    # Sayaç bilgilerini göster
    cv2.putText(frame, f"Giris: {entry_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cikis: {exit_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Icerdeki Kisi: {total_people}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Video işleme döngüsü
cv2.namedWindow('Dukkan Giris-Cikis Sayaci')
#cv2.setMouseCallback('Dukkan Giris-Cikis Sayaci', RGB)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Görüntüyü küçült
    frame = cv2.resize(frame, (1280, 720))
    
    # YOLO ile nesne tespiti
    results = model.track(frame, persist=True, conf=0.5, iou=0.3, classes=0, verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()  # x, y, width, height
        track_ids = results[0].boxes.id.int().cpu().tolist()  # tracking IDs
        
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
                    "in_area": False,
                    "direction": None
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
                person_tracks[track_id]["direction"] = crossing
            
            # Kişinin rengini belirle
            if person_tracks[track_id]["in_area"]:
                color = (0, 255, 255)  # Sarı - alan içinde
            elif person_tracks[track_id]["counted"]:
                color = (0, 255, 0) if person_tracks[track_id]["direction"] == "entry" else (0, 0, 255)
            else:
                color = (255, 255, 255)  # Beyaz - alan dışında
            
            # Çizgiden yeterince uzaklaşınca sayım durumunu sıfırla
            if abs(cy - (area1[0][1] + area1[2][1]) / 2) > 100:
                person_tracks[track_id]["counted"] = False
            
            # Son konumu güncelle
            person_tracks[track_id]["last_x"] = cx
            person_tracks[track_id]["last_y"] = cy
            
            # Kişiyi çerçeve içine al
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    draw_areas(frame)
    cv2.imshow("Dukkan Giris-Cikis Sayaci", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()