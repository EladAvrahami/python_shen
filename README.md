# python_shen
python class

<pre> 
  sudo raspi-config


  import cv2
import mediapipe as mp
import google.generativeai as genai

# הגדרת מודל Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# הגדרת זיהוי אובייקטים
mp_object_detection = mp.solutions.object_detection
detector = mp_object_detection.ObjectDetection()

# הפעלת מצלמה
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # זיהוי אובייקטים
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            label = detection.label
            print(f"Detected: {label}")

            # שליחת מידע ל-Gemini לניתוח
            response = genai.chat(f"What can you tell me about {label}?")
            print("Gemini Response:", response.text)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






  bash: python3 object_detection.py

<pre\>
