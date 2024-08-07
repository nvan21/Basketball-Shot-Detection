from ultralytics import YOLO
import cv2
import math

vid = cv2.VideoCapture("basketball_shooting.mp4")
model = YOLO("basketball_v2.pt")

# fmt:off
class_names = ["basketball"]
# fmt:on

while True:
    success, img = vid.read()

    if success is False:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the bounding box confidence and round it to 2 digits
            conf = math.ceil(box.conf[0] * 100) / 100

            # Get the class name
            cls = int(box.cls[0])
            current_class = class_names[cls]

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

    cv2.imshow("Image", img)
    cv2.waitKey(0)
