import cv2
cap = cv2.VideoCapture(0)
status, img = cap.read()

face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #make picture gray
faces = face_model.detectMultiScale(gray_picture, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
    face = img[y:y+h, x:x+w] # cut the face frame out
    eyes = eye_model.detectMultiScale(gray_face)

for (ex,ey,ew,eh) in eyes: 
    print(ex, ey, ew, eh)
    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)
print (faces)
print(eyes)
cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
