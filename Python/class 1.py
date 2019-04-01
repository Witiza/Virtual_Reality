import  numpy as np
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080))
while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        out.write(frame)

        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

