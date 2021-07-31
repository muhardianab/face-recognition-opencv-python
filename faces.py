import numpy as np
import cv2
import pickle

# load model 
face_cascade = cv2.CascadeClassifier('/home/muhardianab/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# load labels trained
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    ori_labels = pickle.load(f)
    labels = {v:k for k,v in ori_labels.items()}

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=50)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]   #(ycoord-start, ycoord-end)
        #roi_color = frame[y:y+h, x:x+w]

        # recognize face
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_, labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        
        # save frame last detection
        img_item = "last-detection.png"
        cv2.imwrite(img_item, frame) #(img_item, roi_gray)

        # labeled object while recognize
        color = (255, 0, 0)     #BGR 0-255
        stroke = 2
        weight = x + w      #end coord x
        height = y + h      #end coord y
        cv2.rectangle(frame, (x, y), (weight, height), color, stroke)

    # show window frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()