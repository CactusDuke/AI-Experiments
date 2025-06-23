import cv2
import numpy as np

face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_model)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_model)
            for eye in eyes:
                if eye is not None:
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def show_eyes(img, face_model, eye_model):
    gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make picture gray
    faces = face_model.detectMultiScale(gray_picture, 1.3, 5)

    for (x,y,w,h) in faces:
        gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
        face = img[y:y+h, x:x+w] # cut the face frame out
        eyes = eye_model.detectMultiScale(gray_face)

    for (ex,ey,ew,eh) in eyes: 
        cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)
def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints

def cut_eyebrows(img):
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)return img
    return img


def detect_eyes(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame, 1.3, 5) # detect eyes
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,255),2) #Drawing eyes

        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye

def detect_faces(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame

def nothing(x):
    pass

main()