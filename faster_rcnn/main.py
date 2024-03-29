import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import time

class TrafficSignDetection:

    def __init__(self, model_path):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.to(self.device)

        self.classes = { 1: "Ahead_only",
                2: "Beware_of_ice_snow",
                3: "Bicycles_crossing",
                4: "Bumpy_road",
                5: "Children_crossing",
                6: "Dangerous_curve_to_the_left",
                7: "Dangerous_curve_to_the_right",
                8: "Double_curve",
                9: "End_of_all_speed_and_passing_limits",
                10: "End_of_no_passing",
                11: "End_of_no_passing_by_vehicles_over_3_5_metric_tons",
                12: "End_of_speed_limit_80_km_h",
                13: "General_caution",
                14: "Go_straight_or_left",
                15: "Go_straight_or_right",
                16: "Keep_left",
                17: "Keep_right",
                18: "No_entry",
                19: "No_passing",
                20: "No_passing_for_vehicles_over_3_5_metric_tons",
                21: "No_vehicles",
                22: "Pedestrians",
                23: "Priority_road",
                24: "Right_of_way_at_the_next_intersection",
                25: "Road_narrows_on_the_right",
                26: "Road_work",
                27: "Roundabout_mandatory",
                28: "Slippery_road",
                29: "Speed_limit_100_km_h",
                30: "Speed_limit_120_km_h",
                31: "Speed_limit_20_km_h",
                32: "Speed_limit_30_km_h",
                33: "Speed_limit_50_km_h",
                34: "Speed_limit_60_km_h",
                35: "Speed_limit_70_km_h",
                36: "Speed_limit_80_km_h",
                37: "Stop",
                38: "Traffic_signals",
                39: "Turn_left_ahead",
                40: "Turn_right_ahead",
                41: "Vehicles_over_3_5_metric_tons_prohibited",
                42: "Wild_animals_crossing",
                43: "Yield"}

    def detect_traffic_signs(self, frame):

        image = Image.fromarray(frame).convert("RGB")
        image_tensor = T.ToTensor()(image).to(self.device)
        with torch.no_grad():
            prediction = self.model([image_tensor])

        return prediction

    def plot_traffic_signs(self, frame, prediction):

        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score > 0.6:
                box = box.astype(np.int32)
                frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                frame = cv2.putText(frame, f"{self.classes[label]}: {score:.2f}", (box[0], box[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame


if __name__ == "__main__":

    detector = TrafficSignDetection("train1000.pkl")

    cap = cv2.VideoCapture("Video1.mp4")
    fps_start_time = time.time()
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        prediction = detector.detect_traffic_signs(frame)
        frame = detector.plot_traffic_signs(frame, prediction)

        cv2.imshow('Traffic Signs Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_end_time = time.time()
    fps = frame_count / (fps_end_time - fps_start_time)
    print(f"Average FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()