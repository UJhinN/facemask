import cv2 
import math
import numpy as np

def detect_eye(img):
    # Load the Cascade Classifier for eye detection
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    # Detect eyes in the image
    eyes_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=9)
    eyes = []

    # Iterate through detected eyes and store them in a list
    for (x, y, w, h) in eyes_rect:
        eyes.append((x, y, w, h))
     
    degree = None
    diff_size_eyes = None
    w_0, h_0, w_1, h_1 = 0, 0, 0, 0  

    # If two eyes are detected
    if len(eyes) == 2:
        left_eye = eyes[1]
        right_eye = eyes[0]
        if eyes[0][0] <= eyes[1][0]:
            left_eye = eyes[0]
            right_eye = eyes[1]

        x_0, y_0, w_0, h_0 = left_eye
        x_1, y_1, w_1, h_1 = right_eye

        # Calculate the angle of the face using the positions of the eyes
        if x_0 - x_1 != 0:
            tan_theta = (y_0 - y_1) / (x_0 - x_1)
        else:
            tan_theta = 0

        theta = math.atan(tan_theta)
        degree = math.degrees(theta)

        # Calculate the size difference between the eyes
        diff_size_eyes = (w_0 - w_1) + (h_0 - h_1)

    return degree, img, diff_size_eyes

def detect_face(img, mask, ratio_w_h_mask):
    # Load the Cascade Classifier for face detection
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    # Detect faces in the image
    faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=9)

    for (x, y, w, h) in faces_rect:
        # Detect eyes and find the angle of the face
        degree, img_detection_eye, diff_size_eyes = detect_eye(img)
        # Apply the mask to adjust the face
        img = image_transform(x, y, w, h, img_detection_eye, mask, ratio_w_h_mask, degree, diff_size_eyes)
    return img 

def image_transform(x, y, w, h, frame, img, ratio_w_h_mask, degree, diff_size_eyes):
    # Resize the mask to fit the face
    h = round(h * ratio_w_h_mask)
    img = cv2.resize(img, (w, h))
    
    # Rotate the face to match the angle of the detected face
    if degree is not None:
        M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), -degree, 1)
        img = cv2.warpAffine(img, M, (w, h))

    # Transform the shape of the mask to fit the face
    if diff_size_eyes is not None:
        matrix = None
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        if diff_size_eyes < 0:
            diff_size_eyes = abs(diff_size_eyes) 
            pts2 = np.float32([[0, diff_size_eyes], [w, 0], [0, h - diff_size_eyes], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
        elif diff_size_eyes >= 0:
            diff_size_eyes = abs(diff_size_eyes)
            pts2 = np.float32([[0, 0], [w - diff_size_eyes, 0], [0, h], [w - diff_size_eyes, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)

        if matrix is not None:
            img = cv2.warpPerspective(img, matrix, (w, h))

    # Capture the face from the circular region around the face
    roi = frame[y:y+h, x:x+w]
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the color from the face image to black and white to use as a mask
    ret, mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Combine the face image with the mask
    frame_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(img, img, mask=mask)
    dst = cv2.add(frame_bg, img_fg)
    frame[y:y+h, x:x+w] = dst
    return frame

if __name__ == "__main__":
    # Load the mask image
    mask = cv2.imread('D:\\Y1\\StudyY1\\facemask\\mask.png')
    w, h, channel = mask.shape
    ratio_w_h_mask = w / h
    # Start the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        # Detect faces and adjust with the mask
        img = detect_face(frame, mask, ratio_w_h_mask)
        # Show the result
        cv2.imshow("Face Mask", img)

        # Stop when ESC is pressed
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
