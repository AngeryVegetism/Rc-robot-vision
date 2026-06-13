from ultralytics import YOLO
import easyocr
import cv2

# Load models
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0) # put here the camera

print("Press Q to quit")

while True:
    ret, frame = cap.read()  # <-- reads one frame from the stream
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, verbose=False)  # <-- pass frame, not cap

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        crop = frame[y1:y2, x1:x2]  # <-- crop from frame, not cap

        text = reader.readtext(crop, detail=0)
        print(f"Detected text: {text}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # <-- draw on frame
        cv2.putText(frame, str(text), (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + OCR", frame)  # <-- show frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()