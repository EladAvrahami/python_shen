# python_shen
python class
https://ai.google.dev/gemini-api/docs/quickstart?hl=he&lang=python
<pre> 




bash: python3 object_detection.py

import cv2
import mediapipe as mp
import google.generativeai as genai

# הגדרת מפתח API של Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# הגדרת זיהוי אובייקטים באמצעות MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# יצירת זיהוי פנים
detector = mp_face_detection.FaceDetection(model_selection=0)

# הפעלת מצלמה
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # המרת הפריים למדיהפייפ
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    # עיבוד תוצאות הזיהוי
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

            # שליחת נתונים ל-Gemini לקבלת מידע על הפנים
            response = genai.chat("Describe a human face.")
            cv2.putText(frame, response.text[:50], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # הצגת התמונה
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



  ***********************************
>>> %Run object_detection.py
Traceback (most recent call last):
  File "/home/eladron/Desktop/gemini_project/object_detection.py", line 10, in <module>
    mp_object_detection = mp.solutions.object_detection
AttributeError: module 'mediapipe.python.solutions' has no attribute 'object_detection'
>>> 

**************************************************
import cv2
import numpy as np
import mediapipe as mp
import google.generativeai as genai

# הגדרת מפתח API של Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# הגדרת זיהוי אובייקטים עם MediaPipe Solutions
mp_object_detection = mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils

detector = mp_object_detection.ObjectDetection(model_selection=0)

# הפעלת מצלמה
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # המרת הפריים למדיהפייפ
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    # עיבוד תוצאות הזיהוי
    if results.detections:
        for detection in results.detections:
            # ציור מסגרת סביב האובייקט
            mp_drawing.draw_detection(frame, detection)

            # זיהוי האובייקט ושליחת מידע ל-Gemini
            label = detection.label_id if detection.label_id else "Unknown"
            response = genai.chat(f"Describe {label} briefly.")
            cv2.putText(frame, response.text[:50], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # הצגת התמונה
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

*****************************************************************
  eladron@raspberrypi:~ $ python3
Python 3.11.2 (main, Nov 30 2024, 21:22:50) [GCC 12.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mediapipe as mp
>>> print(dir(mp.tasks))
['BaseOptions', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'audio', 'components', 'genai', 'text', 'vision']


********************************************************************
  source ~/Desktop/gemini_project/my_env/bin/activate
pip show mediapipe
***********************************
>>> %Run object_detection.py
Traceback (most recent call last):
  File "/home/eladron/Desktop/gemini_project/object_detection.py", line 5, in <module>
    from mediapipe.tasks import vision
ImportError: cannot import name 'vision' from 'mediapipe.tasks' (/home/eladron/Desktop/gemini_project/my_env/lib/python3.11/site-packages/mediapipe/tasks/__init__.py)
>>>




    ****************************************************************************************************
import cv2
import numpy as np
import mediapipe as mp
import google.generativeai as genai
from mediapipe.tasks import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# הגדרת מפתח API של Gemini
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

    # המרת פריים למדיהפייפ
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    # עיבוד תוצאות הזיהוי
    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            label = detection.categories[0].category_name if detection.categories else "Unknown"

            # יצירת מסגרת סביב האובייקט
            x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # שליחת האובייקט ל-Gemini לקבלת תיאור
            response = genai.chat(f"Describe {label} briefly.")
            cv2.putText(frame, response.text[:50], (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # הצגת התוצאה על המסך
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




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
import cv2#elad
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


   

  
  
