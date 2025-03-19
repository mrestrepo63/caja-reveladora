import cv2
import numpy as np
import pixy
from ctypes import *
from pixy import BlockArray, Pixy2CCC

# Initialize Pixy2
pixy.init()
pixy.change_prog("video")

# Function to capture frames from Pixy2
def get_frame():
    frame = pixy.video_get_frame(0, 0, 320, 200, None)
    if frame is not None:
        return np.frombuffer(frame, dtype=np.uint8).reshape((200, 320, 1))
    return None

while True:
    # Capture image from Pixy2
    frame = get_frame()
    if frame is None:
        continue
    
    # Convert to BGR for OpenCV processing
    imagen = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Convert to HSV color space
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate bright areas (fluorescent regions)
    lower_skin = np.array([0, 0, 50], dtype=np.uint8)
    upper_skin = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations
    kernel = np.ones((5, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Extract hand region
    hand_only = cv2.bitwise_and(imagen, imagen, mask=mask)
    gray_hand = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)
    gray_hand = cv2.equalizeHist(gray_hand)
    blurred = cv2.GaussianBlur(gray_hand, (5, 5), 0)

    # Threshold for fluorescence detection
    _, fluorescence = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
    
    # Find and draw contours
    contours, _ = cv2.findContours(fluorescence, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagen, contours, -1, (0, 255, 0), 2)

    # Display results
    cv2.imshow('Silueta de la Mano', gray_hand)
    cv2.imshow('Detección de Fluorescencia', fluorescence)
    cv2.imshow('Contornos de Fluorescencia', imagen)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
