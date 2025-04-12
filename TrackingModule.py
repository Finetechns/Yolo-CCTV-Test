import math
import time

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
                # Update position
                self.center_points[matched_id] = (cx, cy)
                # Update last seen
                self.last_seen[matched_id] = current_time
                # Update movement history
                if matched_id not in self.movement_history:
                    self.movement_history[matched_id] = []
                self.movement_history[matched_id].append((cx, cy))
                if len(self.movement_history[matched_id]) > 5: 
                    self.movement_history[matched_id].pop(0)
                objects_bbs_ids.append([x, y, w, h, matched_id])
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

    def check_crossing(self, object_id, cy, line_position, direction='down'):
        if object_id in self.movement_history and len(self.movement_history[object_id]) >= 2:
            prev_cy = self.movement_history[object_id][-2][1]
            
            if direction == 'down':
                if prev_cy < line_position and cy >= line_position and object_id not in self.has_crossed:
                    self.has_crossed.add(object_id)
                    return True
            else:  # direction == 'up'
                if prev_cy > line_position and cy <= line_position and object_id not in self.has_crossed:
                    self.has_crossed.add(object_id)
                    return True
        
        return False 