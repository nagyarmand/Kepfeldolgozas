import cv2
import numpy as np
import math
from sort import Sort
import cvzone



class YOLODetector:
    def __init__(self, model_path="yolov8x.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        results = self.model(frame_region, stream=True)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])

                if cls in [2, 3, 5, 7]:
                    detections.append([x1, y1, x2, y2])

        return detections


class CustomDetectorBase:
    def __init__(self, history=100, varThreshold=50, detectShadows=True):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=varThreshold, detectShadows=detectShadows
        )

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
            if 80000 > area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, processed_mask

class CustomDetector1(CustomDetectorBase):
    def __init__(self):
        super().__init__(history=500, varThreshold=75, detectShadows=True)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        grey = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 3)
        subtracted_mask = self.bg_subtractor.apply(blur)
        _, processed_mask = cv2.threshold(subtracted_mask, 1, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)


        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80000 > area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, processed_mask



class CustomDetector2A(CustomDetectorBase):
    def __init__(self):
        super().__init__(history=100, varThreshold=100, detectShadows=False)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        grey = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 3)
        subtracted_mask = self.bg_subtractor.apply(blur)
        processed_mask = cv2.dilate(subtracted_mask, np.ones((7, 7)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        processed_mask = cv2.erode(processed_mask, np.ones((15, 15)), iterations=1)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80000 > area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, processed_mask

class CustomDetector2B(CustomDetectorBase):
    def __init__(self):
        super().__init__(history=300, varThreshold=25, detectShadows=True)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        subtracted_mask = self.bg_subtractor.apply(frame_region)
        _, processed_mask = cv2.threshold(subtracted_mask, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80000 > area > 2500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, processed_mask

class CustomDetector3(CustomDetectorBase):
    def __init__(self):
        super().__init__(history=400, varThreshold=7, detectShadows=True)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        subtracted_mask = self.bg_subtractor.apply(frame_region)
        _, processed_mask = cv2.threshold(subtracted_mask, 254, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=5)


        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80000 > area > 1200:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, processed_mask

class CustomDetector4(CustomDetectorBase):
    def __init__(self):
        super().__init__(history=450, varThreshold=5, detectShadows=True)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        grey = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)

        subtracted_mask = self.bg_subtractor.apply(grey)
        _, processed_mask = cv2.threshold(subtracted_mask, 1, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)


        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80000 > area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, processed_mask

class CustomDetector5(CustomDetectorBase):
    def __init__(self):
        super().__init__(history=200, varThreshold=75, detectShadows=False)

    def detect(self, frame, mask):
        frame_region = cv2.bitwise_and(frame, mask)
        grey = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(grey)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 3)
        subtracted_mask = self.bg_subtractor.apply(blur)

        _, processed_mask = cv2.threshold(subtracted_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50000 > area > 5000:
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
                    if dist < 25:
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
    def __init__(self, video_source, mask_image, detector, tracker, limits1, limits2=None, limits3=None, cy_range=None,
                 counter_position1=(10, 485), counter_position2=(1125, 510), counter_position3=(600, 520)):
        self.cap = cv2.VideoCapture(video_source)
        self.mask = cv2.imread(mask_image)
        self.detector = detector
        self.tracker = tracker
        self.limits1 = limits1
        self.limits2 = limits2
        self.limits3 = limits3
        self.cy_range = cy_range
        self.counter_position1 = counter_position1
        self.counter_position2 = counter_position2
        self.counter_position3 = counter_position3
        self.count1 = []
        self.count2 = []
        self.count3 = []

    def process_frame(self, frame):
        if isinstance(self.detector, CustomDetectorBase):  # Ha CustomDetector
            detections, processed_mask = self.detector.detect(frame, self.mask)
        else:  # YOLODetector esetén
            detections = self.detector.detect(frame, self.mask)
            processed_mask = None

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
            cv2.putText(frame, f'ID: {tracking_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if self.limits1[0] < cx < self.limits1[2] and self.limits1[1] - 10 < cy < self.limits1[3] + 10 and tracking_id not in self.count1:
                    self.count1.append(tracking_id)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

            if self.limits2 and  self.limits2[0] < cx < self.limits2[2] and self.limits2[1] - 10 < cy < self.limits2[3] + 10 and tracking_id not in self.count2:
                    self.count2.append(tracking_id)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

            if self.limits3 and self.limits3[0] < cx < self.limits3[2] and self.limits3[1] - 10 < cy < self.limits3[3] + 10 and tracking_id not in self.count3:
                    self.count3.append(tracking_id)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

        return frame, processed_mask

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, processed_mask = self.process_frame(frame)

            if processed_mask is not None:
                cv2.imshow("Mask", processed_mask)

            cv2.line(frame, (self.limits1[0], self.limits1[1]), (self.limits1[2], self.limits1[3]), (0, 0, 255), 5)
            cvzone.putTextRect(frame, f'{len(self.count1):03}', self.counter_position1, scale=2, thickness=2, offset=10, colorR=(15, 15, 15))
            if self.limits2:
                cv2.line(frame, (self.limits2[0], self.limits2[1]), (self.limits2[2], self.limits2[3]), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{len(self.count2):03}', self.counter_position2, scale=2, thickness=2, offset=10, colorR=(15, 15, 15))
            if self.limits3:
                cv2.line(frame, (self.limits3[0], self.limits3[1]), (self.limits3[2], self.limits3[3]), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{len(self.count3):03}', self.counter_position3, scale=2, thickness=2,offset=10, colorR=(15, 15, 15))


            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                return None
            elif key == ord('1'):
                return 1
            elif key == ord('2'):
                return 2
            elif key == ord('3'):
                return 3
            elif key == ord('4'):
                return 4
            elif key == ord('5'):
                return 5


        self.cap.release()
        cv2.destroyAllWindows()


class Application:
    def __init__(self, detector_type, tracker_type):
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.current_video = 1

        if detector_type == "YOLO":
            tracker1 = Sort(max_age=35, min_hits=5, iou_threshold=0.4) if tracker_type == "SORT" else EuclideanTracker()
            tracker2 = Sort(max_age=40, min_hits=5, iou_threshold=0.4) if tracker_type == "SORT" else EuclideanTracker()
            tracker3 = Sort(max_age=50, min_hits=4, iou_threshold=0.3) if tracker_type == "SORT" else EuclideanTracker()
            tracker4 = Sort(max_age=50, min_hits=4, iou_threshold=0.05) if tracker_type == "SORT" else EuclideanTracker()
            tracker5 = Sort(max_age=50, min_hits=4, iou_threshold=0.05) if tracker_type == "SORT" else EuclideanTracker()



            detector = YOLODetector()
            self.video1_processor = VideoProcessor(
                "media/video1.mp4", "media/mask1.png", detector, tracker1,
                limits1=[0, 275, 1280, 275], counter_position1=(10, 285)
            )
            self.video2_processor = VideoProcessor(
                "media/video2.mp4", "media/mask2.png", detector, tracker2,
                limits1=[80, 475, 580, 475], limits2=[690, 500, 1114, 500],
                counter_position1=(10, 485), counter_position2=(1125, 510)
            )
            self.video3_processor = VideoProcessor(
                "media/video3.mp4", "media/mask3.png", detector, tracker3,
                limits1=[85, 400, 590, 400], limits2=[710, 500, 1200, 500],
                counter_position1=(10, 410), counter_position2=(1200, 510)
            )
            self.video4_processor = VideoProcessor(
                "media/video4.mp4", "media/mask4.png", detector, tracker4,

                limits1 = [180, 525, 650, 525], limits2 = [750, 500, 1174, 500],
                counter_position1 = (110, 535), counter_position2 = (1180, 510)
            )
            self.video5_processor = VideoProcessor(
                "media/video5.mp4", "media/mask5.png", detector, tracker5,
                limits1=[120, 350, 500, 350], limits2=[500, 550, 780, 550], limits3=[850, 550, 1150, 550],
                counter_position1=(50, 360), counter_position2=(465, 560), counter_position3=(1150, 560)
            )

        else:
            tracker1 = Sort(max_age=35, min_hits=5, iou_threshold=0.4) if tracker_type == "SORT" else EuclideanTracker()
            tracker2 = Sort(max_age=25, min_hits=2, iou_threshold=0.5) if tracker_type == "SORT" else EuclideanTracker()
            tracker3 = Sort(max_age=40, min_hits=3, iou_threshold=0.4) if tracker_type == "SORT" else EuclideanTracker()
            tracker4 = Sort(max_age=50, min_hits=4, iou_threshold=0.05) if tracker_type == "SORT" else EuclideanTracker()
            tracker5 = Sort(max_age=30, min_hits=3, iou_threshold=0.05) if tracker_type == "SORT" else EuclideanTracker()


            self.video1_processor = VideoProcessor(
                "media/video1.mp4", "media/mask1.png", CustomDetector1(), tracker1,
                limits1=[0, 400, 1280, 400], counter_position1=(10, 410)
            )
            self.video2_processor = VideoProcessor(
                "media/video2.mp4", "media/mask2.png", CustomDetector2A(), tracker2,
                limits1=[80, 550, 580, 550], limits2=[690, 500, 1114, 500],
                counter_position1=(10, 560), counter_position2=(1125, 510)
            )
            self.video3_processor = VideoProcessor(
                "media/video3.mp4", "media/mask3CUST.png", CustomDetector3(), tracker3,
                limits1=[85, 450, 590, 450], limits2=[710, 500, 1200, 500],
                counter_position1=(10, 460), counter_position2=(1200, 510)
            )
            self.video4_processor = VideoProcessor(
                "media/video4.mp4", "media/mask4CUST.png", CustomDetector4(), tracker4,
                limits1 = [180, 525, 650, 525], limits2 = [750, 500, 1174, 500],
                counter_position1 = (110, 535), counter_position2 = (1180, 510)
            )
            self.video5_processor = VideoProcessor(
                "media/video5.mp4", "media/mask5.png", CustomDetector5(), tracker5,
                limits1=[120, 450, 500, 450], limits2=[500, 550, 780, 550], limits3=[850, 550, 1150, 550],
                counter_position1=(50, 460), counter_position2=(465, 560), counter_position3=(1150, 560)
            )


    def run(self):
        while True:
            next_video = None

            if self.current_video == 1:
                next_video = self.video1_processor.run()
            elif self.current_video == 2:
                next_video = self.video2_processor.run()
            elif self.current_video == 3:
                next_video = self.video3_processor.run()
            elif self.current_video == 4:
                next_video = self.video4_processor.run()
            elif self.current_video == 5:
                next_video = self.video5_processor.run()

            if next_video is None:
                break
            elif next_video in [1, 2, 3, 4, 5]:
                self.current_video = next_video


if __name__ == "__main__":
    detector_type = "Custom"  # "YOLO" vagy "Custom"
    tracker_type = "SORT"  # "SORT" vagy "Euclidean"
    app = Application(detector_type, tracker_type)
    app.run()
