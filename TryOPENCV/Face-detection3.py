'''
For This code, I really want to enter the file name for do the face detection
But I can not fix code. Maybe I have to change the IDE

This is actually work on Google COlab
'''


import cv2

# Load the pre-trained face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory path
directory_path = "r\"C:\\Users\\Pannawit\\Documents\\GitHub\\Deep-learning\\TryOPENCV\\"

# Get the file name from the user
file_name = input("Enter the file name of the image (including extension): ")

# Combine directory path and file name
file_path = directory_path + file_name + ".jpg"

# Load the image
image = cv2.imread(file_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with face detection
cv2.imshow("Face Detection", image)
cv2.waitKey(0)  # Wait for any key press
cv2.destroyAllWindows()  # Close all OpenCV windows
