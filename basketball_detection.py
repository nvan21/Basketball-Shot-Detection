from ultralytics import YOLO
from ultralytics.engine.results import Results
from sklearn.linear_model import LinearRegression
import cv2
import numpy as np
import math


class BasketballDetector:
    def __init__(self, model: YOLO, fps: int, class_names: list[str]):
        self.model = model
        self.fps = fps
        self.class_names = class_names
        self.ball_positions = []
        self.ball_trajectory = []
        self.person_positions = []
        self.backboard_bbox = []

        self.has_shot = False
        self.confidence = []

    def predict(self, img: np.ndarray) -> Results:
        return self.model(img, stream=True)

    def plot_parabolic_trajectory(self, img: np.ndarray):
        ball_positions = np.array(self.ball_positions)
        x = ball_positions[:, 0]
        y = ball_positions[:, 1]
        coeffs = np.polyfit(ball_positions[:, 0], ball_positions[:, 1], 2)
        a, b, c = coeffs

        x_fit = np.linspace(0, 2 * max(x), 1000)
        y_fit = a * x_fit**2 + b * x_fit + c

        for x, y in zip(x_fit, y_fit):
            cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0), -1)

    def is_shot_going_in(self, img: np.ndarray) -> bool:
        going_in = False
        ball_radius = 12 / 2
        x, y, w, h = self.backboard_bbox
        rx = int(x + 27 * self.scale)
        ry = int(y + 30 * self.scale)
        rw = int(w * 18 / 72)
        rh = int(h * 5 / 42)
        x1, y1 = rx, ry
        x2, y2 = rx + rw, ry + rh
        cv2.rectangle(
            img,
            (int(x1 + ball_radius), y1),
            (int(x2 - ball_radius), y2),
            (0, 255, 0),
            3,
        )

        ball_trajectory = np.array(self.ball_trajectory)
        mask = (
            (x1 + ball_radius <= ball_trajectory[:, 0])
            & (ball_trajectory[:, 0] <= x2 - ball_radius)
            & (y1 <= ball_trajectory[:, 1])
            & (ball_trajectory[:, 1] <= y2)
        )

        if mask.sum() >= 15:
            going_in = True

        self.confidence.append(going_in)
        score = sum(self.confidence) / len(self.confidence)

        cv2.putText(
            img,
            f"Going In: {going_in}",
            (550, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    def draw_boxes(self, results: Results, img: np.ndarray) -> list:
        has_sports_ball = False
        has_person = False

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x3, y3 = int((x1 + x2) / 2), int((y1 + y2) / 2)
                r = int((x2 - x1) / 2)

                # Get the bounding box confidence and round it to 2 digits
                conf = math.ceil(box.conf[0] * 100) / 100

                # Get the class name
                cls = int(box.cls[0])
                current_class = self.class_names[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 255))
                cv2.putText(
                    img,
                    f"{current_class} {conf}",
                    (max(0, x1), max(20, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )
                if current_class == "sports ball":
                    has_sports_ball = True
                    self.ball_positions.append((x3, y3, r))

                if current_class == "person":
                    has_person = True
                    self.person_positions.append((x1, y1, x2, y2))

                # Only check to see if the person has shot if the person and ball are in the same frame
                # and the person has not already shot
                if has_sports_ball and has_person:
                    person_position = self.person_positions[-1]
                    ball_position = self.ball_positions[-1]
                    self.has_shot = self.detect_shooting(
                        person_position=person_position, ball_position=ball_position
                    )

    def reset_shot_stats(self):
        self.confidence = []

    def plot_physics_trajectory(
        self,
        img: np.ndarray,
        frame_height: int,
        frame_width: int,
        v0x: float,
        v0y: float,
        num_points: int = 1000,
        g: float = -386.04,  # in
    ):
        self.ball_trajectory = []
        g = g * self.scale
        x0, y0 = self.ball_positions[-1][0], self.ball_positions[-1][1]

        t_values = np.linspace(0, 1.5, num=num_points)

        for t in t_values:
            x = x0 + v0x * t
            y = y0 + v0y * t - 0.5 * g * t**2
            self.ball_trajectory.append((x, y))

            if 0 <= int(x) < frame_width and 0 <= int(y) < frame_height:
                cv2.circle(
                    img, (int(x), int(y)), 2, (255, 0, 0), -1
                )  # Blue dots for predicted path

    # TODO: Extracting the backboard is myopic because it requires white stripes on the backboard and the right camera angle. Fix this by adding object detection
    def extract_backboard(self, image: np.ndarray):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the white color range in HSV
        lower_white = np.array([0, 0, 200], dtype=np.uint8)  # Lower bound for white
        upper_white = np.array([180, 55, 255], dtype=np.uint8)  # Upper bound for white

        # Create a mask for white colors
        mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to track the best candidate for the backboard
        best_backboard = None
        max_area = 0

        for contour in contours:
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio and area
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)

            # Define expected aspect ratio and area range for the backboard
            expected_aspect_ratio_range = (
                1,
                2,
            )  # Example aspect ratio range for a rectangular backboard
            expected_area_range = (
                0,
                100000,
            )  # Example area range for the backboard in pixels

            # Check if the contour matches the expected properties
            if (
                expected_aspect_ratio_range[0]
                < aspect_ratio
                < expected_aspect_ratio_range[1]
                and expected_area_range[0] < area < expected_area_range[1]
            ):

                # Update the best backboard candidate if this one is larger
                if area > max_area:
                    max_area = area
                    best_backboard = (x, y, w, h)

        # Draw the detected backboard on the original image
        if best_backboard is not None:
            self.backboard_bbox = best_backboard
            x, y, w, h = best_backboard
            cv2.rectangle(
                image, (x, y), (x + w, y + h), (255, 0, 0), 3
            )  # Draw a blue rectangle for the backboard
            cv2.putText(
                image,
                "Backboard",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        return (w, h)

    def set_scale(self, backboard_w: int, backboard_h: int):
        reg_backboard_height = 42  # in
        reg_backboard_width = 72  # in

        # Calculate the scale (units of pixels/in)
        self.scale = (
            (backboard_w / reg_backboard_width) + (backboard_h / reg_backboard_height)
        ) / 2
        self.scale = backboard_w / reg_backboard_width

    def calculate_initial_velocity(self):
        if len(self.ball_positions) < 3:
            return None, None

        # Use the last three points to fit a linear model
        time_intervals = np.array([(i + 1) for i in range(3)]) / self.fps
        x_coords = np.array([p[0] for p in self.ball_positions[-3:]])
        y_coords = np.array([p[1] for p in self.ball_positions[-3:]])

        # Fit linear regression models for x and y coordinates
        x_model = LinearRegression().fit(time_intervals.reshape(-1, 1), x_coords)
        y_model = LinearRegression().fit(time_intervals.reshape(-1, 1), y_coords)

        # Initial velocities are the slopes of the fitted lines
        v0x = x_model.coef_[0]
        v0y = y_model.coef_[0]
        return v0x, v0y

    def detect_shooting(self, person_position: tuple, ball_position: tuple) -> bool:
        """Detects whether or not the ball has left the person's hands. It returns true if the ball has and false if the ball hasn't

        Args:
            person_position (tuple): _description_
            ball_position (tuple): _description_

        Returns:
            bool: _description_
        """
        # Unpack the position coordinates
        x1, y1, x2, y2 = person_position
        cx, cy, r = ball_position

        # Calculate the closest point on the rectangle to the circle's center
        closest_x = max(x1, min(cx, x2))
        closest_y = max(y1, min(cy, y2))

        # Calculate the distance from the circle's center to this closest point
        distance_x = cx - closest_x
        distance_y = cy - closest_y

        # Calculate the square of the distance
        distance_squared = distance_x**2 + distance_y**2

        # Check if the distance is less than or equal to the circle's radius squared
        return not (distance_squared <= r**2)
