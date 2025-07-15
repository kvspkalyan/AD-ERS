from detection import load_model, predict_frame
from emergency_response import trigger_emergency
import cv2

model = load_model()
class_names = ['Accident', 'Normal']  # match order in training!

cap = cv2.VideoCapture("videos/test2.mp4")
streak = 0
threshold = 0.98
required_streak = 5
triggered = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    label, conf = predict_frame(model, frame, class_names)
    print(f"{label} ({conf:.2f})")

    if label == "Accident" and conf > threshold:
        streak += 1
    else:
        streak = 0

    if streak >= required_streak and not triggered:
        trigger_emergency()
        triggered = True

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
