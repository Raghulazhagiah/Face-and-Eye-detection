import cv2

Eye_Classifier = cv2.CascadeClassifier('raw.githubusercontent.com_murtazahassan_OpenCV-Python-Tutorials-and-Projects_master_Intermediate_Custom Object Detection_haarcascades_haarcascade_eye.xml')
Face_Classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    #_, img = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #grayy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = Eye_Classifier.detectMultiScale(frame,1.1,4)
    face = Face_Classifier.detectMultiScale(frame,1.1,4)

    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),4)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),6)
    cv2.imshow("Eyes",frame)
    #cv2.imshow("Face",img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cap.release()
