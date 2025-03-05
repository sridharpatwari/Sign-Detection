import cv2
import os
import mediapipe as mp

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Function to draw skeleton using landmarks
def draw_hand_skeleton(image_path, output_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    results = hands.process(img_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections between them (the skeleton)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Save the result with the skeleton drawn
        cv2.imwrite(output_path, img)
        print(f"Skeleton drawn and saved at: {output_path}")
    else:
        print(f"No hands detected in: {image_path}")

# Directory paths
input_dir = "C:\\Users\\joshi\\Sign-Language-To-Text-Conversion\\images1"
output_dir = "C:\\Users\\joshi\\Sign-Language-To-Text-Conversion\\Skeleton_Images"

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
                
                # Draw skeleton and save the image
                draw_hand_skeleton(input_image_path, output_image_path)

# Release resources
hands.close()