import cv2
import os
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Function to draw skeleton using landmarks on a white image
def draw_hand_skeleton_on_white(image_path, output_path, white_image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    results = hands.process(img_rgb)

    # Load the blank white page for skeleton drawing
    white_img = cv2.imread(white_image_path)

    if white_img is None:
        print(f"Error loading white image: {white_image_path}")
        return

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Iterate through landmarks and draw the lines connecting them
            for i in range(21):  # Hand has 21 landmarks
                # Get the position of the landmark
                x, y = int(hand_landmarks.landmark[i].x * white_img.shape[1]), int(hand_landmarks.landmark[i].y * white_img.shape[0])

                # Draw the landmarks as small circles (optional)
                cv2.circle(white_img, (x, y), 5, (0, 255, 0), -1)  # Green circle

            # Draw the hand connections using green lines
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]

                # Get the position of the two connected landmarks
                start_coords = int(start.x * white_img.shape[1]), int(start.y * white_img.shape[0])
                end_coords = int(end.x * white_img.shape[1]), int(end.y * white_img.shape[0])

                # Draw a green line between them
                cv2.line(white_img, start_coords, end_coords, (0, 255, 0), 2)

        # Save the result with skeleton drawn
        cv2.imwrite(output_path, white_img)
        print(f"Skeleton drawn and saved at: {output_path}")
    else:
        print(f"No hands detected in: {image_path}")

# Directory paths
input_dir = "C:\\Users\\joshi\\dataset\\asl_dataset\\asl_dataset"
output_dir = "C:\\Users\\joshi\\Sign-Language-To-Text-Conversion\\AtoZ_3.1"
white_image_path = "C:\\Users\\joshi\\Sign-Language-To-Text-Conversion\\white.jpg"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process specific folders (A-Z) inside the Gray_Images folder
for letter_folder in os.listdir(input_dir):
    letter_folder_path = os.path.join(input_dir, letter_folder)
    
    # Process only the directories (A-Z) that exist in the input path
    if os.path.isdir(letter_folder_path):
        output_subfolder_path = os.path.join(output_dir, letter_folder)
        
        # Create corresponding output folder for the letter if it doesn't exist
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)

        # Process all images (with .jpeg/.png/.jpg) in the letter subfolder
        for filename in os.listdir(letter_folder_path):
            # Consider only image files (check extension)
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                input_image_path = os.path.join(letter_folder_path, filename)
                output_image_path = os.path.join(output_subfolder_path, f"skeleton_{filename}")
                
                # Draw skeleton and save the image with skeleton on a white background
                draw_hand_skeleton_on_white(input_image_path, output_image_path, white_image_path)

# Release resources
hands.close()