import cv2
import os
import numpy as np

# Initialize webcam
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Webcam not initialized.")
    exit()

# Paths for saving images
base_path = "C:\\Users\\joshi\\Sign-Language-To-Text-Conversion\\images1\\"
os.makedirs(base_path, exist_ok=True)

# Gesture and counter setup
gesture_label = "A"  # Starting with A
count = 0
capture_flag = False  # Toggle to capture images
offset = 30  # Padding for ROI around the hand

print("Press 'a' to start/stop capturing images.")
print("Press 'n' to move to the next gesture.")
print("Press 'Esc' to exit.")

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally
    cv2.putText(
        frame, f"Gesture: {gesture_label}, Images: {count}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    
    # Display the webcam feed
    cv2.imshow("Webcam Feed", frame)
    key = cv2.waitKey(1)

    # Exit when 'Esc' is pressed
    if key == 27:
        break

    # Toggle capturing with 'a'
    elif key & 0xFF == ord('a'):
        capture_flag = not capture_flag
        print(f"Capture {'started' if capture_flag else 'stopped'} for gesture {gesture_label}.")
    
    # Move to the next gesture with 'n'
    elif key & 0xFF == ord('n'):
        gesture_label = chr((ord(gesture_label) + 1 - ord('A')) % 26 + ord('A'))  # Cycle through A-Z
        count = 0  # Reset count for the new gesture
        print(f"Now capturing for gesture: {gesture_label}")
    
    # Save images if capturing is enabled
    if capture_flag:
        # Define a region of interest (ROI) based on frame size
        height, width, _ = frame.shape
        x_center, y_center = width // 2, height // 2
        roi_size = 200  # ROI width and height
        x_start, y_start = x_center - roi_size // 2, y_center - roi_size // 2
        roi = frame[y_start:y_start + roi_size, x_start:x_start + roi_size]

        # Ensure the folder exists for the current gesture
        gesture_folder = os.path.join(base_path, gesture_label)
        os.makedirs(gesture_folder, exist_ok=True)

        # Save the image
        file_name = f"{gesture_label.lower()}{count}.jpg"
        file_path = os.path.join(gesture_folder, file_name)
        cv2.imwrite(file_path, roi)
        print(f"Saved: {file_path}")

        count += 1

capture.release()
cv2.destroyAllWindows()