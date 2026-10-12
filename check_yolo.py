
from services.face import detect_faces, draw_face_boxes
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not working!")
    raise SystemExit

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    faces = detect_faces(frame)
    output = draw_face_boxes(frame, faces)

    cv2.imshow("YOLO Face Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
