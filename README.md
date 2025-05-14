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

# הגדרת מודל Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# הגדרת זיהוי אובייקטים
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="efficientdet_lite0.tflite"),
    running_mode=RunningMode.IMAGE
)

detector = ObjectDetector.create_from_options(options)

# הפעלת מצלמה
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    if results.detections:
        for detection in results.detections:
            label = detection.categories[0].category_name if detection.categories else "Unknown"
            print(f"Detected: {label}")

            response = genai.chat(f"What can you tell me about {label}?")
            print("Gemini Response:", response.text)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



<pre\>

 <pre> 
Traceback (most recent call last):
  File "/home/eladron/Desktop/gemini_project/object_detection.py", line 20, in <module>
    detector = ObjectDetector.create_from_options(options)
  File "/home/eladron/Desktop/gemini_project/my_env/lib/python3.11/site-packages/mediapipe/tasks/python/vision/object_detector.py", line 238, in create_from_options
    return cls(
  File "/home/eladron/Desktop/gemini_project/my_env/lib/python3.11/site-packages/mediapipe/tasks/python/vision/core/base_vision_task_api.py", line 70, in __init__
    self._runner = _TaskRunner.create(graph_config, packet_callback)
RuntimeError: Unable to open file at /home/eladron/Desktop/gemini_project/efficientdet_lite0.tflite
>>> 

Traceback (most recent call last):
  File "/home/eladron/Desktop/Object_detection.py", line 9, in <module>
    mp_object_detection = mp.solutions.object_detection
AttributeError: module 'mediapipe.python.solutions' has no attribute 'object_detection'
>>> 

   
<pre\>
  
  
