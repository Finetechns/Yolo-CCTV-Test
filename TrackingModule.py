import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

class EuclideanDistTracker:
    def __init__(self, dist_threshold=30, disappear_threshold=10, min_detection_confidence=0.4):
        self.center_points = {}
        self.id_count = 0
        self.dist_threshold = dist_threshold
        self.last_seen = {}
        self.disappear_threshold = disappear_threshold
        self.min_detection_confidence = min_detection_confidence
        self.movement_history = {}
        self.has_crossed = set()
        self.entry_count = 0
        self.exit_count = 0
        self.total_people = 0
        self.area1 = [(748, 498), (741, 470), (986, 394), (1026, 430)]  # İlk alan için koordinatlar
        self.area2 = [(748, 498), (741, 470), (986, 394), (1026, 430)]  # İkinci alan için koordinatlar

    def update(self, objects_rect, confidences=None):
        objects_bbs_ids = []
        current_time = time.time()

        self._clean_disappeared_objects(current_time)

        for i, rect in enumerate(objects_rect):
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            if confidences and confidences[i] < self.min_detection_confidence:
                continue

            same_object_detected = False
            matched_id = None

            for obj_id in self.center_points:
                if obj_id in self.movement_history:
                    prev_positions = self.movement_history[obj_id]
                    if len(prev_positions) >= 2:
                        pred_x = prev_positions[-1][0] + (prev_positions[-1][0] - prev_positions[-2][0])
                        pred_y = prev_positions[-1][1] + (prev_positions[-1][1] - prev_positions[-2][1])
                        dist_to_pred = math.hypot(cx - pred_x, cy - pred_y)
                        
                        if dist_to_pred < self.dist_threshold * 1.5:  
                            matched_id = obj_id
                            same_object_detected = True
                            break

            if not same_object_detected:
                min_dist = float('inf')
                for obj_id, pt in self.center_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = obj_id

                if min_dist < self.dist_threshold:
                    same_object_detected = True

            if same_object_detected:
                self.center_points[matched_id] = (cx, cy)
                self.last_seen[matched_id] = current_time
                if matched_id not in self.movement_history:
                    self.movement_history[matched_id] = []
                self.movement_history[matched_id].append((cx, cy))
                if len(self.movement_history[matched_id]) > 5: 
                    self.movement_history[matched_id].pop(0)
                objects_bbs_ids.append([x, y, w, h, matched_id])
                
                # Alan geçiş kontrolü
                self._check_area_crossing(matched_id, cx, cy)
            else:
                self.center_points[self.id_count] = (cx, cy)
                self.last_seen[self.id_count] = current_time
                self.movement_history[self.id_count] = [(cx, cy)]
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        new_movement_history = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
            if object_id in self.movement_history:
                new_movement_history[object_id] = self.movement_history[object_id]

        self.center_points = new_center_points.copy()
        self.movement_history = new_movement_history.copy()
        return objects_bbs_ids

    def _clean_disappeared_objects(self, current_time):
        ids_to_remove = []
        for obj_id in self.last_seen:
            if current_time - self.last_seen[obj_id] > self.disappear_threshold:
                ids_to_remove.append(obj_id)

        for obj_id in ids_to_remove:
            self.center_points.pop(obj_id, None)
            self.last_seen.pop(obj_id, None)
            self.movement_history.pop(obj_id, None)
            self.has_crossed.discard(obj_id)

    def _check_area_crossing(self, object_id, cx, cy):
        if object_id in self.movement_history and len(self.movement_history[object_id]) >= 2:
            prev_cx, prev_cy = self.movement_history[object_id][-2]
            
            # Alan kontrolü
            in_area1 = cv2.pointPolygonTest(np.array(self.area1), (cx, cy), False) >= 0
            in_area2 = cv2.pointPolygonTest(np.array(self.area2), (cx, cy), False) >= 0
            was_in_area1 = cv2.pointPolygonTest(np.array(self.area1), (prev_cx, prev_cy), False) >= 0
            was_in_area2 = cv2.pointPolygonTest(np.array(self.area2), (prev_cx, prev_cy), False) >= 0
            
            in_area = in_area1 or in_area2
            was_in_area = was_in_area1 or was_in_area2
            
            if not was_in_area and in_area:
                if cy < prev_cy:
                    self.entry_count += 1
                    self.total_people += 1
                else:
                    self.exit_count += 1
                    self.total_people = max(0, self.total_people - 1)
            elif was_in_area and not in_area:
                if cy < prev_cy:
                    self.exit_count += 1
                    self.total_people = max(0, self.total_people - 1)
                else:
                    self.entry_count += 1
                    self.total_people += 1

    def draw_areas(self, frame):
 
        cv2.polylines(frame, [np.array(self.area1)], True, (0, 255, 255), 2)
        cv2.polylines(frame, [np.array(self.area2)], True, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Giris: {self.entry_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Cikis: {self.exit_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Icerdeki Kisi: {self.total_people}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main():
    stream_url = "https://hls.ibb.gov.tr/ls/cam_turistik/b_kapalicarsi.stream/chunklist.m3u8"
    cap = cv2.VideoCapture(stream_url)
    model = YOLO('yolov8s.pt')
    tracker = EuclideanDistTracker()

    cv2.namedWindow('Dukkan Giris-Cikis Sayaci')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (1280, 720))
        
        results = model.track(frame, persist=True, conf=0.5, iou=0.3, classes=0, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            detections = []
            for box in boxes:
                x, y, w, h = box
                detections.append([int(x - w/2), int(y - h/2), int(w), int(h)])
            
            boxes_ids = tracker.update(detections)
            
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        tracker.draw_areas(frame)
        cv2.imshow("Dukkan Giris-Cikis Sayaci", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 