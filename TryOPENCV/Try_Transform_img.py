import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\Pannawit\Documents\GitHub\Deep-learning\TryOPENCV\Monkey.jpg")

cv2.imshow("Normal", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# First Try
image_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("90", image_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Second Try
rows, cols, _ = image.shape
# 3 by 3 matrix as it is required for the OpenCV library, don't worry about the details of it for now.
M = np.float32([[1, 0.2, 0],
                [-0.1, 1, 45], 
                [0, 0, 1]])
image_rotated_sheared = cv2.warpPerspective(image, M, (int(cols), int(rows)))

Text = "W T F"
org = (60, 300)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 5
color = (255, 255, 255)
thickness = 5

cv2.putText(image_rotated_sheared, Text, org, font, font_scale, color, thickness, cv2.LINE_AA)


cv2.imshow("WTF", image_rotated_sheared)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Try Third
# Define a color adjustment matrix for increasing brightness
brightness_matrix = np.array([[99, -99, 0],
                              [0, 1.2, 0],
                              [0, 0, 1.2]])

# Apply the color adjustment
brightened_image = cv2.transform(image, brightness_matrix)

cv2.imshow("Brightened Image", brightened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

