# Basketball Shot Detection

Need a way to calculate the initial position and velocity of the ball. Since the bounding box of the ball is inside the bounding box of the person (for the most part), then
the ball can be considered as leaving the person's hand when it first leaves the bounding box of the person. The center coordinates of the ball in that frame will be considered the initial position. The center coordinates of the ball in the frame the next time that the ball is detected will be used to calculate the intial velocity. From there, the trajectory can be plotted

# Structure

eval.py -> Evaluates the video and controls the training loop
utils.py -> Preprocesses the video
basketball_detection.py -> Has BasketballDetector class which takes in a frame, uses YOLO for inference, and then returns a
