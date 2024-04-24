import cv2
import requests
from PIL import Image
import numpy as np
import mediapipe as mp
from io import BytesIO

# Initialize MediaPipe drawing and pose estimation solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class helper:

    @staticmethod 
    def downloadImageFromUrl(url):
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            print(f"Error downloading image from {url}")
            return None

    @staticmethod
    def getImageFromPath(path):
        frame = cv2.imread(path)
        return frame
    
    @staticmethod
    def resizeImage(image, screen_width, screen_height):
        if image is None:
            return None
        # Calculate the desired width and height for the clothing image
        clothing_width = int(screen_width / 3.3)
        clothing_height = int(screen_height / 2.5)

        # Resize the clothing image
        resized_img = image.resize((clothing_width, clothing_height), Image.LANCZOS)
        return resized_img

    @staticmethod
    def jpgToPng(image):
        return image.convert('RGBA')

# Replace with your preferred pose estimation library (e.g., OpenPose)
def estimate_pose(frame):
    # Convert BGR image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a new instance of the pose detector
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_landmarks:
        # Process the image with the pose detector
        results = pose_landmarks.process(frame_rgb)

    # Extract keypoints from MediaPipe results
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            height, width, _ = frame.shape
            keypoint = (int(landmark.x * width), int(landmark.y * height))
            keypoints.append(keypoint)

    return keypoints

def overlay_image(background, overlay, keypoints, vêtement_size, clothing_type, is_photo=True):
    # Check if keypoints are detected
    if len(keypoints) < 3:
        print("Not enough keypoints detected. Showing cloth in the middle.")
        # Calculate the position to center the cloth image in the middle of the frame
        overlay_x = (background.width - overlay.width) // 2
        overlay_y = (background.height - overlay.height) // 2
    else:
        # Calculate the division point for upper and lower clothing placement
        if is_photo:
            division_point = int(background.height * 0.58) if clothing_type == 'upper' else int(background.height * 0.90)
        else:
            division_point = int(background.height * 0.80) if clothing_type == 'upper' else int(background.height * 0.90)
        
        # Ignore the first 11 keypoints, which correspond to the face
        body_keypoints = keypoints[7:]

        if not body_keypoints:
            print("No body keypoints detected. Showing cloth in the middle.")
            # Calculate the position to center the cloth image in the middle of the frame
            overlay_x = (background.width - overlay.width) // 2
            overlay_y = (background.height - overlay.height) // 2
        else:
            # Calculate the center of the body using keypoints for shoulders
            center_x = sum(keypoint[0] for keypoint in body_keypoints) // len(body_keypoints)
            center_y = sum(keypoint[1] for keypoint in body_keypoints) // len(body_keypoints)

            # Calculate the position for overlaying the cloth
            overlay_x = max(0, center_x - overlay.width // 2)
            overlay_y = max(0, center_y - overlay.height // 2)

            # Adjust position based on clothing type
            if clothing_type == 'upper':
                overlay_y = min(division_point - overlay.height, overlay_y)
            else:
                overlay_y = min(int((background.height - overlay.height)), int(overlay_y*1.3))

    # Convert overlay to RGBA if not already
    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')

    # Overlay the cloth on the background image
    background.paste(overlay, (overlay_x, overlay_y), overlay)

def capture_video(vêtement_img, clothing_type , vêtement_size):
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Inside the while loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform pose estimation (replace with library function call)
        keypoints = estimate_pose(frame)

        # Convert OpenCV frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Overlay vêtement on frame (basic placement)
        overlay_image(frame_pil, vêtement_img, keypoints, vêtement_size, clothing_type, is_photo=False)

        # Convert PIL image back to OpenCV frame
        frame_rgb = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow('Virtual Try-On', frame_rgb)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Modify the try_on_photo function definition
def try_on_photo(photo, vêtement_img, vêtement_size, clothing_type, is_photo=True):
    # Use the provided photo directly

    # Perform pose estimation (replace with library function call)
    keypoints = estimate_pose(photo)

    # Convert OpenCV frame to PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))

    # Overlay vêtement on frame (basic placement)
    overlay_image(frame_pil, vêtement_img, keypoints, vêtement_size, clothing_type, is_photo=True)

    # Convert PIL image back to OpenCV frame
    frame_rgb = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('Virtual Try-On', frame_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return frame_rgb
\
def choose_clothing_image():
    vêtement_url = input("Enter the URL of the clothing image: ")
    return vêtement_url

def resize_photo_to_screen(photo_path, screen_width, screen_height):
    # Read the photo
    frame = helper.getImageFromPath(photo_path)

    # Resize the photo to fit the screen dimensions while maintaining aspect ratio
    aspect_ratio = frame.shape[1] / frame.shape[0]
    if aspect_ratio > screen_width / screen_height:
        # Resize based on width
        resized_frame = cv2.resize(frame, (screen_width, int(screen_width / aspect_ratio)))
    else:
        # Resize based on height
        resized_frame = cv2.resize(frame, (int(screen_height * aspect_ratio), screen_height))

    return resized_frame

def imageHandler(screen_height ,photo_path , vêtement_url, screen_width, clothing_type):
    # photo_path = input("Enter the path to the photo: ")
    resized_photo = resize_photo_to_screen(photo_path, screen_width, screen_height)

    # Ask user for clothing image URL
    # vêtement_url = choose_clothing_image()
    resized_clothing_img = helper.resizeImage( helper.downloadImageFromUrl(vêtement_url) , screen_width, screen_height)
    
    # Calculate the size of the clothing image
    vêtement_size = (resized_clothing_img.width, resized_clothing_img.height)

    try_on_photo( resized_photo , resized_clothing_img , vêtement_size , clothing_type , is_photo=True )
    
def videoHandler(screen_height , screen_width , clothing_type):
    # Ask user for clothing image URL
    vêtement_url = choose_clothing_image()
    resized_clothing_img = helper.resizeImage(helper.downloadImageFromUrl(vêtement_url), screen_width, screen_height)
    
    # Calculate the size of the clothing image
    vêtement_size = (resized_clothing_img.width, resized_clothing_img.height)

    capture_video(resized_clothing_img, clothing_type, vêtement_size) 

import json
def greet(name): # for testing
    return("Hello, " + name)

def main(input_type ,photo_path,dress_url, screen_width = 800 , screen_height = 800 , clothing_type = 'upper'):
    if input_type == 'photo':
        # Ask user for photo path
        imageHandler(screen_height ,photo_path, dress_url , screen_width , clothing_type)
    elif input_type == 'live' :
        # Ask user for upper or lower body clothing
       videoHandler(screen_height , screen_width , clothing_type)
    else:
        return "Invalid input type. enum: ('photo' or 'live') "