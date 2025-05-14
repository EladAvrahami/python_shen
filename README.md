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


  #mediapipe problome solution : 
  
  import mediapipe as mp

mp_object_detection = mp.tasks.vision.ObjectDetector
detector = mp_object_detection.create_from_options(mp_object_detection.ObjectDetectorOptions())

  ***********************************************************************************************************
  mp_object_detection = mp.tasks.vision.ObjectDetector

options = mp_object_detection.ObjectDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="efficientdet_lite0.tflite"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
detector = mp_object_detection.create_from_options(options)
************************************************************************************************

  
>>> %Run object_detection.py
Traceback (most recent call last):
  File "/home/eladron/yolo/object_detection.py", line 2, in <module>
    import mediapipe as mp
ModuleNotFoundError: No module named 'mediapipe'
>>> 
  *type object 'ObjectDetector' has no attribute 'ObjectDetectorOptions'
**********************************
pip install google-generativeai opencv-python mediapipe
pip install numpy pillow


888888888888888888888888888888888888888888888888888888888888
    ![image](https://github.com/user-attachments/assets/ffd1cfa2-116f-40d5-b39f-a4fd0466d705)
    

   pip install google-generativeai opencv-python mediapipe numpy pillow


    pip list | grep "google-generativeai\|opencv-python\|mediapipe\|numpy\|pillow"


google-generativeai          0.8.5
mediapipe                    0.10.18
numpy                        1.26.4
opencv-python                4.11.0.86
pillow                       11.2.1

    ********************************************************************************************
import cv2
import numpy as np
import mediapipe as mp
import google.generativeai as genai
from mediapipe.tasks import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# הגדרת מודל Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# הגדרת זיהוי אובייקטים
options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="efficientdet_lite0.tflite"),
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.ObjectDetector.create_from_options(options)

# הפעלת מצלמה
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # המרת הפריים לתמונה שמדיהפייפ יכול לעבד
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    # הצגת תוצאות זיהוי האובייקטים
    if results.detections:
        for detection in results.detections:
            label = detection.categories[0].category_name if detection.categories else "Unknown"
            print(f"Detected: {label}")

            # שליחת מידע ל-Gemini לניתוח וקבלת תגובה
            response = genai.chat(f"What can you tell me about {label}?")
            print("Gemini Response:", response.text)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



    
<pre\>


   

  
  
