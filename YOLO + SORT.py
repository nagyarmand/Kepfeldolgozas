
import cvzone
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from sort import Sort  # SORT tracker importálása


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
        detections = np.empty((0, 5))  # SORT kompatibilis formátum (x1, y1, x2, y2, confidence)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0])

                if cls in [2, 3, 5, 7] and conf > 0.4:  # Detect relevant classes
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        return detections


class VideoProcessor:
    def __init__(self, video_source, region_mask, tracker, limits, additional_limits=None):
        self.capture = cv.VideoCapture(video_source)
        self.region_mask = region_mask
        self.tracker = tracker
        self.limits = limits
        self.additional_limits = additional_limits  # Másik irányú vonal koordinátái
        self.count = []
        self.additional_count = []  # Másik irányú számláló

    def process_frame(self, frame, object_detector):
        detections = object_detector.detect_objects(frame, self.region_mask)
        results_tracer = self.tracker.update(detections)  # SORT frissítése

        # Kirajzolja a számláló vonalakat pirosan alapértelmezetten
        cv.line(frame, (self.limits[0][0], self.limits[0][1]),
                (self.limits[0][2], self.limits[0][3]), (0, 0, 255), 5)
        if self.additional_limits:
            cv.line(frame, (self.additional_limits[0], self.additional_limits[1]),
                    (self.additional_limits[2], self.additional_limits[3]), (0, 0, 255), 5)

        for result in results_tracer:
            x1, y1, x2, y2, tracking_id = map(int, result)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Kirajzolja az autók köré a corner rect-et
            cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=2, colorR=(255, 0, 0))
            # Kirajzolja a középpontjukat egy piros ponttal
            cv.circle(frame, (cx, cy), 4, (0, 0, 255), cv.FILLED)

            # Ellenőrzés, hogy az objektum átlépte-e az első vonalat
            if (self.limits[0][0] < cx < self.limits[0][2] and
                    self.limits[0][1] - 10 < cy < self.limits[0][3] + 20 and
                    tracking_id not in self.count):
                self.count.append(tracking_id)
                # Villanjon fel a középpont zölden
                cv.circle(frame, (cx, cy), 6, (0, 255, 0), cv.FILLED)

            # Ellenőrzés, hogy az objektum átlépte-e a második vonalat
            if self.additional_limits and \
                    (self.additional_limits[0] < cx < self.additional_limits[2] and
                     self.additional_limits[1] - 10 < cy < self.additional_limits[1] + 20 and
                     tracking_id not in self.additional_count):
                self.additional_count.append(tracking_id)
                # Villanjon fel a középpont zölden
                cv.circle(frame, (cx, cy), 6, (0, 255, 0), cv.FILLED)

        return frame


class Application:
    def __init__(self):
        self.object_detector = ObjectDetector('yolov8x.pt')
        self.tracker1 = Sort(max_age=200, min_hits=4, iou_threshold=0.3)  # SORT használata
        self.tracker2 = Sort(max_age=200, min_hits=4, iou_threshold=0.3)

        # Video1 feldolgozó, amely két vonalat figyel
        self.video1_processor = VideoProcessor(
            "media/videoplayback.mp4",
            cv.imread("media/mask.png"),
            self.tracker1,
            [[83, 475, 580, 475]],
            additional_limits=[690, 500, 1114, 500]
        )
        # Video2 feldolgozó, amely egy vonalat figyel
        self.video2_processor = VideoProcessor(
            "media/video1.mp4",
            cv.imread("media/mask2.png"),
            self.tracker2,
            [[0, 400, 1280, 400]]
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
                    print("Video 1 vége")
                    break
                frame = self.video1_processor.process_frame(frame, self.object_detector)
                # Kirajzolja az első vonal számlálóját
                cvzone.putTextRect(frame, f'{len(self.video1_processor.count):03}', (10, 485),
                                   scale=2, thickness=2, offset=10, colorR=(15, 15, 15))
                # Kirajzolja a második vonal számlálóját
                cvzone.putTextRect(frame, f'{len(self.video1_processor.additional_count):03}', (1125, 510),
                                   scale=2, thickness=2, offset=10, colorR=(15, 15, 15))

            elif self.current_video == 2:
                ret, frame = self.video2_processor.capture.read()
                if not ret:
                    print("Video 2 vége")
                    break
                frame = self.video2_processor.process_frame(frame, self.object_detector)
                # Kirajzolja az egyetlen vonal számlálóját
                cvzone.putTextRect(frame, f'{len(self.video2_processor.count):03}', (10, 405),
                                   scale=2, thickness=2, offset=10, colorR=(15, 15, 15))

            if frame is not None:  # Csak akkor jelenítjük meg, ha van frame
                cv.imshow('Video', frame)

            key = cv.waitKey(1)
            if key == 27:  # Escape key to exit
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
