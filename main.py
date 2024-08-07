from basketball_detection import BasketballDetector
from ultralytics import YOLO
import cv2
from utils import CLASS_NAMES

# Create YOLO model
model_path = "yolov8l.pt"
model = YOLO(model_path)

# Create video feed
vid_path = "basketball_shooting/three_front_facing_three_point.mov"
vid = cv2.VideoCapture(vid_path)

# Parameters
fps = 60
scale = 0.25
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)


# Create basketball detector
ball_detector = BasketballDetector(model=model, fps=fps, class_names=CLASS_NAMES)

while True:
    success, img = vid.read()
    if success is False:
        break

    img = cv2.resize(img, (width, height))

    results = ball_detector.predict(img)
    ball_detector.draw_boxes(results=results, img=img)
    w, h = ball_detector.extract_backboard(image=img)
    ball_detector.set_scale(backboard_w=w, backboard_h=h)

    if len(ball_detector.ball_positions) >= 3 and ball_detector.has_shot:
        v0x, v0y = ball_detector.calculate_initial_velocity()
        ball_detector.plot_physics_trajectory(
            img=img, frame_height=height, frame_width=width, v0x=v0x, v0y=v0y
        )
        # ball_detector.plot_parabolic_trajectory(img)
        ball_detector.is_shot_going_in(img)
    else:
        ball_detector.reset_shot_stats()

    cv2.imshow("Basketball Shot Predictor", img)
    cv2.waitKey(1)