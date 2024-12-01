import cv2
import numpy as np
import math
import cvzone


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
                if dist < 50:  # Threshold for detecting same object
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
    def __init__(self, video_source, mask_image, tracker):
        self.cap = cv2.VideoCapture(video_source)
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=80, detectShadows=True)
        self.region = cv2.imread(mask_image)
        self.tracker = tracker

    def process_frame(self, frame):
        detections = []
        if self.region is not None:  # Ellenőrzi, hogy a maszk érvényes-e
            frame_region = cv2.bitwise_and(frame, self.region)
        else:
            frame_region = frame

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
                detections.append([x, y, w, h])

        return detections, mask

    def release(self):
        self.cap.release()


class Application:
    def __init__(self):
        self.tracker1 = Tracker()
        self.tracker2 = Tracker()

        self.video1_processor = VideoProcessor("media/videoplayback.mp4", "media/mask.png", self.tracker1)
        self.video2_processor = VideoProcessor("media/video1.mp4", "media/mask2.png", self.tracker2)

        self.current_video = 1
        self.count1_1 = []
        self.count1_2 = []
        self.count2 = []

    def run(self):
        while True:
            if self.current_video == 1:
                self.process_video1()
            elif self.current_video == 2:
                self.process_video2()

            key = cv2.waitKey(5)
            if key == 27:  # ESC to exit
                break
            elif key == ord('1'):
                self.current_video = 1
            elif key == ord('2'):
                self.current_video = 2

        self.cleanup()

    def process_video1(self):
        limits1 = [83, 475, 580, 475]
        limits2 = [690, 500, 1114, 500]

        ret, frame = self.video1_processor.cap.read()
        if not ret:
            print("Video 1 vége")
            return

        detections, mask = self.video1_processor.process_frame(frame)

        cv2.line(frame, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 5)
        cv2.line(frame, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 5)

        boxes_ids = self.video1_processor.tracker.update(detections)

        for box_id in boxes_ids:
            x, y, w, h, tracking_id = box_id
            cx, cy = x + w // 2, y + h // 2

            # Rajzolja a corner rect-et és a piros vonalat, ha cy 400 és 600 között van
            if 400 < cy < 600:
                cvzone.cornerRect(frame, (x, y, w, h), l=5, t=2, colorR=(255, 0, 0))
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), cv2.FILLED)

            if limits1[0] < cx < limits1[2] and limits1[1] - 10 < cy < limits1[3] + 20 and tracking_id not in self.count1_1:
                self.count1_1.append(tracking_id)
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), cv2.FILLED)

            if limits2[0] < cx < limits2[2] and limits2[1] - 10 < cy < limits2[3] + 20 and tracking_id not in self.count1_2:
                self.count1_2.append(tracking_id)
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), cv2.FILLED)

        cvzone.putTextRect(frame, f'{len(self.count1_1):03}', (10, 485), scale=2, thickness=2, offset=10, colorR=(15, 15, 15))
        cvzone.putTextRect(frame, f'{len(self.count1_2):03}', (1125, 510), scale=2, thickness=2, offset=10, colorR=(15, 15, 15))

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

    def process_video2(self):
        ret, frame = self.video2_processor.cap.read()
        if not ret:
            print("Video 2 vége")
            return

        detections, mask = self.video2_processor.process_frame(frame)
        boxes_ids = self.video2_processor.tracker.update(detections)

        cv2.line(frame, (0, 400), (1280, 400), (0, 0, 255), 5)

        for box_id in boxes_ids:
            x, y, w, h, tracking_id = box_id
            cx, cy = x + w // 2, y + h // 2

            # Rajzolja a corner rect-et és a piros vonalat, ha cy 300 és 600 között van
            if 300 < cy < 600:
                cvzone.cornerRect(frame, (x, y, w, h), l=5, t=2, colorR=(255, 0, 0))
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), cv2.FILLED)

            if 370 < cy < 405 and tracking_id not in self.count2:
                self.count2.append(tracking_id)
                # Villanjon a pont zöldre, amikor áthalad a vonalon
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), cv2.FILLED)

        cv2.putText(frame, f'Count: {len(self.count2)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

    def cleanup(self):
        self.video1_processor.release()
        self.video2_processor.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Application()
    app.run()
