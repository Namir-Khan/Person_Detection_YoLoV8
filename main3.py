from ultralytics import YOLO
import cv2
import time

def initialize_camera():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Warm-up the camera
    for i in range(5):
        cap.read()

    return cap

def capture_frame(cap):
    ret, frame = cap.read()

    return frame

def predict_yolov8(model_path, num_frames, cap):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # To keep track of the number of persons detected and detection status
    total_persons_detected = 0
    detection_occurred = False
    
    for _ in range(num_frames):
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform Detection
        results = model(frame, conf=0.4, save=False, show=False)
        
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
        
        # Show Frame
        # cv2.imshow('YOLOv8 Detection', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break
    
    return total_persons_detected, detection_occurred, results


def main():
    model_path = "best.pt"
    num_frames = 1
    num_valid = 3
    cap = initialize_camera()
    total_persons, detection_occurred = predict_yolov8(model_path, num_frames, cap)

    print(f"Total number of persons detected: {total_persons}")
    print(f"Detection occurred: {detection_occurred}")

    # Cleanup
    cap.release()

if __name__ == "__main__" :
    main()