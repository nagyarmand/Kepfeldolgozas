import cv2
import numpy as np
from sort import Sort
import cvzone


class VideoProcessor:
    def __init__(self, video_source, mask_image, limits, additional_limits=None, cy_range=None):
        self.cap = cv2.VideoCapture(video_source)
        self.region = cv2.imread(mask_image)
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=75, detectShadows=True)
        self.tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
        self.limits = limits
        self.additional_limits = additional_limits
        self.cy_range = cy_range
        self.count1 = []
        self.count2 = []

    def process_frame(self, frame):
        detections = []
        frame_region = cv2.bitwise_and(frame, self.region)
        grey = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 3)
        mask = self.object_detector.apply(blur)
        mask = cv2.dilate(mask, np.ones((7, 7)))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 40000 > area > 2500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, x + w, y + h])

        return detections, mask

    def draw_lines(self, frame):
        cv2.line(frame, (self.limits[0], self.limits[1]), (self.limits[2], self.limits[3]), (0, 0, 255), 5)
        if self.additional_limits:
            cv2.line(frame, (self.additional_limits[0], self.additional_limits[1]),
                     (self.additional_limits[2], self.additional_limits[3]), (0, 0, 255), 5)

    def process_detections(self, frame, boxes_ids):
        for box_id in boxes_ids:
            x1, y1, x2, y2, tracking_id = map(int, box_id)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

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

    def run(self):
        is_running = True
        while is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections, mask = self.process_frame(frame)

            detections_for_sort = np.array(detections, dtype=np.float32)
            boxes_ids = self.tracker.update(detections_for_sort) if detections_for_sort.size > 0 else []

            self.draw_lines(frame)
            self.process_detections(frame, boxes_ids)

            cvzone.putTextRect(frame, f'{len(self.count1):03}', (10, 485), scale=2, thickness=2, offset=10,
                               colorR=(15, 15, 15))
            if self.additional_limits:
                cvzone.putTextRect(frame, f'{len(self.count2):03}', (1125, 510), scale=2, thickness=2, offset=10,
                                   colorR=(15, 15, 15))

            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                is_running = False
            elif key == ord('1'):
                return 1
            elif key == ord('2'):
                return 2

        self.cap.release()
        cv2.destroyAllWindows()


class Application:
    def __init__(self):
        self.video1_processor = VideoProcessor(
            "media/video1.mp4",
            "media/mask2.png",
            limits=[0, 400, 1280, 400],
            cy_range=(300, 600)
        )
        self.video2_processor = VideoProcessor(
            "media/videoplayback.mp4",
            "media/mask.png",
            limits=[83, 475, 580, 475],
            additional_limits=[690, 500, 1114, 500],
            cy_range=(400, 600)
        )
        self.current_video = 1

    def run(self):
        while True:
            if self.current_video == 1:
                self.current_video = self.video1_processor.run()
            elif self.current_video == 2:
                self.current_video = self.video2_processor.run()


if __name__ == "__main__":
    app = Application()
    app.run()
