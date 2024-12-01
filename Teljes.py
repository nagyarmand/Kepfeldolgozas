import cv2
import numpy as np
import math
from sort import Sort
import cvzone


class YOLODetector:
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        results = self.model(frame_region, stream=True)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0])

                if cls in [2, 3, 5, 7] and conf > 0.4:  # Autók, motorok, buszok, teherautók
                    detections.append([x1, y1, x2, y2])

        return detections


class CustomDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=75, detectShadows=True)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        grey = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 3)
        subtracted_mask = self.bg_subtractor.apply(blur)
        processed_mask = cv2.dilate(subtracted_mask, np.ones((7, 7)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 40000 > area > 2500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, processed_mask


class EuclideanTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, detections):
        objects_bbs_ids = []

        for rect in detections:
            if len(rect) >= 5:
                x1, y1, x2, y2 = rect[:4]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                same_object_detected = False

                for tracking_id, pt in self.center_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])
                    if dist < 50:  # Távolsági küszöb
                        self.center_points[tracking_id] = (cx, cy)
                        objects_bbs_ids.append([x1, y1, x2, y2, tracking_id])
                        same_object_detected = True
                        break

                if not same_object_detected:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                    self.id_count += 1

        self.center_points = {obj[4]: ((obj[0] + obj[2]) // 2, (obj[1] + obj[3]) // 2) for obj in objects_bbs_ids}
        return objects_bbs_ids


class VideoProcessor:
    def __init__(self, video_source, mask_image, detector, tracker, limits, detector_type, additional_limits=None, cy_range=None):
        self.cap = cv2.VideoCapture(video_source)
        self.mask = cv2.imread(mask_image)
        self.detector = detector
        self.tracker = tracker
        self.detector_type = detector_type
        self.limits = limits
        self.additional_limits = additional_limits
        self.cy_range = cy_range
        self.count1 = []
        self.count2 = []

    def process_frame(self, frame):
        if self.detector_type == "Custom":
            detections, processed_mask = self.detector.detect(frame, self.mask)
        else:
            detections = self.detector.detect(frame, self.mask)
            processed_mask = None  # YOLO nem használ maszkot megjelenítésre

        detections_for_sort = np.array([[x1, y1, x2, y2, 1.0] for (x1, y1, x2, y2) in detections], dtype=np.float32)
        boxes_ids = self.tracker.update(detections_for_sort) if detections_for_sort.size > 0 else []

        for box_id in boxes_ids:
            x1, y1, x2, y2, tracking_id = map(int, box_id)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if self.cy_range and not (self.cy_range[0] < cy < self.cy_range[1]):
                continue

            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=5, t=2, colorR=(255, 0, 0))
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

            if self.limits[0] < cx < self.limits[2] and self.limits[1] - 10 < cy < self.limits[3] + 20:
                if tracking_id not in self.count1:
                    self.count1.append(tracking_id)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

            if self.additional_limits and \
                    self.additional_limits[0] < cx < self.additional_limits[2] and \
                    self.additional_limits[1] - 10 < cy < self.additional_limits[3] + 20:
                if tracking_id not in self.count2:
                    self.count2.append(tracking_id)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

        return frame, processed_mask

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, processed_mask = self.process_frame(frame)

            if self.detector_type == "Custom" and processed_mask is not None:
                cv2.imshow("Mask", processed_mask)

            cv2.line(frame, (self.limits[0], self.limits[1]), (self.limits[2], self.limits[3]), (0, 0, 255), 5)
            if self.additional_limits:
                cv2.line(frame, (self.additional_limits[0], self.additional_limits[1]),
                         (self.additional_limits[2], self.additional_limits[3]), (0, 0, 255), 5)

            cvzone.putTextRect(frame, f'{len(self.count1):03}', (10, 485), scale=2, thickness=2, offset=10,
                               colorR=(15, 15, 15))
            if self.additional_limits:
                cvzone.putTextRect(frame, f'{len(self.count2):03}', (1125, 510), scale=2, thickness=2, offset=10,
                                   colorR=(15, 15, 15))

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                return None
            elif key == ord('1'):
                return 1
            elif key == ord('2'):
                return 2

        self.cap.release()
        cv2.destroyAllWindows()


class Application:
    def __init__(self, detector_type, tracker_type):
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.current_video = 1

        detector = YOLODetector("yolov8x.pt") if self.detector_type == "YOLO" else CustomDetector()
        tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3) if self.tracker_type == "SORT" else EuclideanTracker()

        self.video1_processor = VideoProcessor(
            "media/video1.mp4",
            "media/mask2.png",
            detector,
            tracker,
            limits=[0, 400, 1280, 400],
            detector_type=self.detector_type,
            cy_range=(300, 600)
        )
        self.video2_processor = VideoProcessor(
            "media/videoplayback.mp4",
            "media/mask.png",
            detector,
            tracker,
            limits=[83, 475, 580, 475],
            detector_type=self.detector_type,
            additional_limits=[690, 500, 1114, 500],
            cy_range=(400, 600)
        )

    def run(self):
        while True:
            next_video = None
            if self.current_video == 1:
                next_video = self.video1_processor.run()
            elif self.current_video == 2:
                next_video = self.video2_processor.run()

            if next_video is None:
                break
            elif next_video in [1, 2]:
                self.current_video = next_video


if __name__ == "__main__":
    detector_type = "Custom"  # "YOLO" vagy "Custom"
    tracker_type = "Euclidean"  # "SORT" vagy "Euclidean"
    app = Application(detector_type, tracker_type)
    app.run()
