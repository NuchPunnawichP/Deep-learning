import cv2
from tkinter import filedialog
import tkinter as tk

# Load the pre-trained face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Ask the user to select a file
file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])

# Check if a file was selected
if not file_path:
    print("No file selected.")
    exit()

# Load the image
image = cv2.imread(file_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with face detection
cv2.imshow("Face Detection", image)
cv2.waitKey(0)  # Wait for any key press
cv2.destroyAllWindows()  # Close all OpenCV windows
