import cv2

image = cv2.imread(r"C:\Users\Pannawit\Documents\GitHub\Deep-learning\TryOPENCV\Monkey.jpg")

image_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow("Hi", image_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()