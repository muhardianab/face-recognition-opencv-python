import cv2

cap = cv2.VideoCapture(0)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    rect, frame = cap.read()
    frame1 = rescale_frame(frame, percent=25)
    cv2.imshow('frame1', frame1)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()