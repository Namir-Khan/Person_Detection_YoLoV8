from ultralytics import YOLO
import cv2
import time

def initialize_camera():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Warm-up the camera
    for i in range(5):
        cap.read()

    # Show frame
    # cv2.imshow('YOLOv8 Detection', frame)
    # if cv2.waitKey(1) == ord('q'):
    #     break

    return cap


def capture_frame(cap):
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Error: Could Not Capture Image")

    return frame


def pred_person(yolo_model, frame):
    # Load the YOLO model
    model = YOLO(yolo_model)

    # Perdict detection
    results = model(frame, conf=0.4, save=False, show=False, verbose=False)

    # Person's Detected
    num_person = results[0].boxes.shape

    # Set Return Variables
    if num_person[0] == 0:
        total_persons_detected = 0
        detection_occurred = False
        # print("Number of Person :",num_person[0])
    else:
        total_persons_detected = num_person[0]
        detection_occurred = True
        # print("Number of Person :",num_person[0])

    return detection_occurred, total_persons_detected


def valid_pred(yolo_model, num_valid, cap):
    # Variable to count number of True detection
    d_count = 0

    # Loop to check mutiple frames
    for _ in range(num_valid):
        # Capture a single frame
        frame = capture_frame(cap)
        detection, total_persons_detected = pred_person(yolo_model, frame)

        # Increment variable
        if detection is True:
            d_count += 1

    # Make prediction
    if d_count > (num_valid/2):
        detection_occurred = True
    else:
        detection_occurred = False

    return detection_occurred, total_persons_detected


def setup(yolo_model, num_valid, cap):
    while True:
        # Give Relay OFF Signal
        print("Relay OFF (Before Prediction)")

        # Prediction
        detection_occurred, total_persons_detected = valid_pred(yolo_model, num_valid, cap)

        if detection_occurred == True:
            # Give Relay ON Signal
            print("Relay ON (After Prediction)")

            # Give Relay OFF after some time Signal
            time.sleep(3)
            print("Relay OFF (After 3 sec)")
        else:
            # Give Relay OFF Signal
            print("Relay OFF (After Prediction)")


def main():
    yolo_model = "best.pt"
    num_valid = 3 # Keep this a small odd number
    cap = initialize_camera()
    setup(yolo_model, num_valid, cap)

    # Cleanup
    cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()