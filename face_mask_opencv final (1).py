import cv2
import numpy as np
import time
import winsound  # for beep (Windows)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened ❌")
    exit()

face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_beep = 0  # for controlling beep timing

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame failed ❌")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_model.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # lower half
        lower_face = face[h//2:h, :]
        lower_gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)

        # simple mask detection (dark area)
        _, thresh = cv2.threshold(lower_gray, 80, 255, cv2.THRESH_BINARY)
        black_ratio = 1 - (cv2.countNonZero(thresh) / thresh.size)

        # ✅ DEFINE LABEL HERE
        if black_ratio > 0.3:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

            # 🔊 Controlled beep (no freeze)
            if time.time() - last_beep > 1:
                winsound.Beep(1000, 300)
                last_beep = time.time()

        # draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # overlay mask area
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y + h//2), (x+w, y+h), color, -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # put label
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()