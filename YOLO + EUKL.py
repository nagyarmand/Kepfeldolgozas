import math
import cvzone
import cv2 as cv
from ultralytics import YOLO



class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def detect_objects(self, frame, region_mask):
        frame_region = cv.bitwise_and(frame, region_mask)
        results = self.model(frame_region, stream=True)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0])

                if cls in [2, 3, 5, 7] and conf > 0.4:  # Detect relevant classes
                    detections.append([x1, y1, x2 - x1, y2 - y1])

        return detections


class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            same_object_detected = False
            for tracking_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:  # Threshold for object association
                    self.center_points[tracking_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, tracking_id])
                    same_object_detected = True
                    break
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()

        return objects_bbs_ids


class VideoProcessor:
    def __init__(self, video_source, region_mask, tracker, limits, additional_limits=None, cy_range=None):
        self.capture = cv.VideoCapture(video_source)
        self.region_mask = region_mask
        self.tracker = tracker
        self.limits = limits
        self.additional_limits = additional_limits
        self.cy_range = cy_range
        self.count = []
        self.additional_count = []

    def process_frame(self, frame, object_detector):
        detections = object_detector.detect_objects(frame, self.region_mask)
        results_tracer = self.tracker.update(detections)

        # Alapértelmezett piros vonalak
        cv.line(frame, (self.limits[0][0], self.limits[0][1]), (self.limits[0][2], self.limits[0][3]), (0, 0, 255), 5)
        if self.additional_limits:
            cv.line(frame, (self.additional_limits[0], self.additional_limits[1]),
                    (self.additional_limits[2], self.additional_limits[3]), (0, 0, 255), 5)

        for result in results_tracer:
            x, y, w, h, tracking_id = result
            cx = x + w // 2
            cy = y + h // 2

            # Csak akkor rajzoljon, ha cy a megadott tartományban van
            if self.cy_range and self.cy_range[0] < cy < self.cy_range[1]:
                cvzone.cornerRect(frame, (x, y, w, h), l=5, t=2, colorR=(255, 0, 0))
                cv.circle(frame, (cx, cy), 4, (0, 0, 255), cv.FILLED)

                # Ha átlépte az első vonalat
                if self.limits[0][0] < cx < self.limits[0][2] and self.limits[0][1] - 10 < cy < self.limits[0][3] + 20:
                    if tracking_id not in self.count:
                        self.count.append(tracking_id)
                        cv.circle(frame, (cx, cy), 6, (0, 255, 0), cv.FILLED)

                # Ha átlépte a második vonalat
                if self.additional_limits and self.additional_limits[0] < cx < self.additional_limits[2] and \
                        self.additional_limits[1] - 10 < cy < self.additional_limits[3] + 20:
                    if tracking_id not in self.additional_count:
                        self.additional_count.append(tracking_id)
                        cv.circle(frame, (cx, cy), 6, (0, 255, 0), cv.FILLED)

        return frame


class Application:
    def __init__(self):
        self.object_detector = ObjectDetector('yolov8x.pt')
        self.tracker1 = Tracker()
        self.tracker2 = Tracker()

        self.video1_processor = VideoProcessor(
            "media/videoplayback.mp4",
            cv.imread("media/mask.png"),
            self.tracker1,
            [[83, 475, 580, 475]],
            additional_limits=[690, 500, 1114, 500],
            cy_range=(400, 600)
        )
        self.video2_processor = VideoProcessor(
            "media/video1.mp4",
            cv.imread("media/mask2.png"),
            self.tracker2,
            [[0, 400, 1280, 400]],
            cy_range=(300, 600)
        )

        self.current_video = 1

    def switch_video(self, key):
        if key == ord('1'):
            self.current_video = 1
        elif key == ord('2'):
            self.current_video = 2

    def run(self):
        while True:
            frame = None  # Alapértelmezett érték a frame-hez
            if self.current_video == 1:
                ret, frame = self.video1_processor.capture.read()
                if not ret:
                    print("Video 1 vége.")
                    break
                frame = self.video1_processor.process_frame(frame, self.object_detector)
                cvzone.putTextRect(frame, f'{len(self.video1_processor.count):03}', (10, 485), scale=2, thickness=2,
                                   offset=10, colorR=(15, 15, 15))
                cvzone.putTextRect(frame, f'{len(self.video1_processor.additional_count):03}', (1125, 510), scale=2,
                                   thickness=2, offset=10, colorR=(15, 15, 15))

            elif self.current_video == 2:
                ret, frame = self.video2_processor.capture.read()
                if not ret:
                    print("Video 2 vége.")
                    break
                frame = self.video2_processor.process_frame(frame, self.object_detector)
                cvzone.putTextRect(frame, f'{len(self.video2_processor.count):03}', (10, 405), scale=2, thickness=2,
                                   offset=10, colorR=(15, 15, 15))

            if frame is not None:  # Csak akkor jelenítjük meg a képkockát, ha van frame
                cv.imshow('Video', frame)

            key = cv.waitKey(1)
            if key == 27:  # ESC to exit
                break
            self.switch_video(key)

        self.cleanup()

    def cleanup(self):
        self.video1_processor.capture.release()
        self.video2_processor.capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    app = Application()
    app.run()

