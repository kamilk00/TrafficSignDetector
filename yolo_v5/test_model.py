import torch
import numpy as np
import cv2 as cv
import time
import os

class TrafficSignsDetectionYOLOv5:

    #initializing a class
    def __init__(self, model):

        self.model = self.load_model(model)
        self.classes = self.model.names        
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\nDevice Used:", self.device)

    #loading YOLOv5 model
    def load_model(self, modelName):      
        return torch.hub.load('yolov5', 'custom', path = modelName, source = 'local')


    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord


    #transform classes to labels
    def class_to_label(self, x):
        return self.classes[int(x)]


    #plot bounding boxes
    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):

            row = cord[i]
            if row[4] >= 0.6:

                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


    def __call__(self):

        cap = cv.VideoCapture("yolov5/Video1.mp4")
        timer = time.time()
        counter = 0

        while True:
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            counter = counter + 1

            if not ret:

                print(f'Average FPS: {(int) (counter / round(time.time() - timer, 4))}')
                break

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)

            cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv.imshow("YOLOv5", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()


if __name__ == '__main__':

    detector = TrafficSignsDetectionYOLOv5(model = 'yolov5/runs/train/gtsdb_all_608_more_img/weights/best.pt')
    detector()