import cv2 as cv
from ultralytics import YOLO
import time
import numpy as np

#load yolov8 model
model = YOLO('runs/detect/gtsdb_all_608_more_img_yolov8s/weights/best.pt')

video_path = "Video1.mp4"
cap = cv.VideoCapture(video_path)
timer = time.time()
counter = 0

while True:

    start_time = time.perf_counter()
    ret, frame = cap.read()

    if not ret:
        print(f'Average FPS: {(int) (counter / round(time.time() - timer, 4))}')
        break

    #predict traffic signs for each frame
    results = model(frame, conf = 0.6)
    annotated_frame = results[0].plot()
    end_time = time.perf_counter()

    fps = 1 / np.round(end_time - start_time, 3)
    cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv.imshow("YOLOv8", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()