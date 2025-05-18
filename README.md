# python_shen
python class
https://ai.google.dev/gemini-api/docs/quickstart?hl=he&lang=python
<pre> 



test code with pro v : 
    import cv2
from google import genai
from google.generativeai import types
import time
import json
import base64
import numpy as np
from PIL import Image as PilImage
import io

# Replace with your actual Gemini API key
GOOGLE_API_KEY = "YOUR_API_KEY"

# Initialize the Gemini Pro Vision model with segmentation capabilities
MODEL_NAME = 'gemini-2.5-pro'  # שנה למודל אחר שתומך בסגמנטציה אם צריך
model = genai.GenerativeModel(model_name=MODEL_NAME, api_key=GOOGLE_API_KEY)

# Open the connection to the camera
cap = cv2.VideoCapture(1)  # Use the correct camera index

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    height, width, _ = frame.shape

    # Capture and process a frame every 1 second (adjust as needed)
    if time.time() % 1 < 0.1:
        _, img_bytes = cv2.imencode('.jpg', frame)
        image = types.Part.from_bytes(
            data=img_bytes.tobytes(),
            mime_type='image/jpeg'
        )
        prompt = """Give the segmentation masks for all prominent items. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels."""

        try:
            response = model.generate_content(contents=[prompt, image])
            if response.text:
                segmentation_results = json.loads(response.text)

                for item in segmentation_results:
                    box_2d_normalized = item["box_2d"]
                    mask_base64 = item["mask"]
                    label = item["label"]

                    # Convert normalized bounding box to original image dimensions
                    ymin = int(box_2d_normalized[0] / 1000 * height)
                    xmin = int(box_2d_normalized[1] / 1000 * width)
                    ymax = int(box_2d_normalized[2] / 1000 * height)
                    xmax = int(box_2d_normalized[3] / 1000 * width)

                    # Decode the base64 encoded PNG mask
                    mask_bytes = base64.b64decode(mask_base64)
                    mask_pil = PilImage.open(io.BytesIO(mask_bytes)).convert('L')  # Grayscale
                    mask_resized = mask_pil.resize((xmax - xmin, ymax - ymin))
                    mask_np = np.array(mask_resized)

                    # Binarize the mask
                    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                    binary_mask_color = np.stack([binary_mask] * 3, axis=-1)  # Convert to BGR

                    # Create a mask for the ROI
                    roi = frame[ymin:ymax, xmin:xmax]

                    # Apply the mask to the ROI
                    masked_roi = cv2.bitwise_and(roi, roi, mask=binary_mask)

                    # Create a color overlay for the mask (e.g., green)
                    color_mask = np.zeros_like(roi, dtype=np.uint8)
                    color_mask[:] = (0, 255, 0)  # Green color
                    colored_mask = cv2.bitwise_and(color_mask, color_mask, mask=binary_mask)

                    # Combine the original ROI with the colored mask
                    alpha = 0.5  # Adjust transparency
                    combined_roi = cv2.addWeighted(roi, 1 - alpha, colored_mask, alpha, 0)

                    # Replace the ROI in the original frame
                    frame[ymin:ymax, xmin:xmax] = combined_roi

                    # Put label text
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print("Raw response:", response.text)
        except Exception as e:
            print(f"Error processing segmentation: {e}")

    # Display the resulting frame with segmentation masks
    cv2.imshow('Live Video with Segmentation', frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()









    
Image segmentation זהות פריטיםו לפלח ולספק מסכה של קווי המתאר שלהם :
    prompt = """Give the segmentation masks for all prominent items. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels."""

    
    

    
 שליחת תמונה captured_frame.jpg לניתוח עי GEMINI וקבלת description from gemini to console 
    from google import genai

client = genai.Client(api_key="")


my_file = client.files.upload(file="captured_frame.jpg")
#/home/eladron/Desktop/gemini_project/
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[my_file, "Caption this image."],)

print(response.text)

    


 

ניסיון צילום תמונה : 
 import cv2

# Open the connection to the camera (try different indexes if 0 doesn't work, e.g., 1, 2)
cap = cv2.VideoCapture(1)  # Use the index that worked for you

# Check if the camera connection was successful
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was captured correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed (Press Space to capture, ESC to exit)', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Spacebar
        # Save the frame as an image file
        cv2.imwrite('captured_frame.jpg', frame)
        print("Image saved as captured_frame.jpg")
        break  # Stop after capturing one image

# Release the capture
cap.release()
cv2.destroyAllWindows()






 

בדיקת קלט ממצלמה וOPENCV 
 import cv2

# Open the connection to the camera (try different indexes if 0 doesn't work, e.g., 1, 2)
cap = cv2.VideoCapture(0)

# Check if the camera connection was successful
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was captured correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Wait for a key press (in milliseconds). 27 is the ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

    
pip install opencv-python google-generativeai Pillow

opencv-python (או בקיצור cv2) היא ספרייה עוצמתית לעיבוד תמונה ווידאו. היא תעזור לנו לגשת למצלמה, ללכוד פריימים ולצייר את הריבועים והכותרות.
google-generativeai היא הספרייה הרשמית של גוגל לעבודה עם מודלי Gemini. היא תאפשר לנו לשלוח בקשות ל-API ולקבל תשובות.
Pillow היא ספרייה נוספת שימושית לעיבוד תמונה.

  18.5.25  
###############################################################################################################################################
    
bash:pip install --upgrade tflite-runtime

import cv2
import mediapipe as mp
import google.generativeai as genai
import tensorflow as tf

# השבתת MLIR Bridge ב-TensorFlow Lite כדי למנוע בעיות תאימות
tf.config.experimental.disable_mlir_bridge()

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

    # בדיקת גודל התמונה לפני עיבוד
    print(f"Frame size: {frame.shape[0]}x{frame.shape[1]}")  # מציג את רוחב וגובה התמונה

    # אם התמונה גדולה מדי או לא תקינה, מבצע שינוי גודל
    if frame.shape[0] == 1 or frame.shape[1] > 32000:
        frame = cv2.resize(frame, (640, 480))
        print("Resized frame to 640x480 to avoid OpenCV errors.")

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

*******************************
    Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1747352130.961954    3562 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
Frame size: 1x921600
Resized frame to 640x480 to avoid OpenCV errors.
************************************************
    image size test :
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

    # בדיקת גודל התמונה לפני עיבוד
    print(f"Frame size: {frame.shape[0]}x{frame.shape[1]}")  # מציג את רוחב וגובה התמונה

    # אם התמונה גדולה מדי, מבצע שינוי גודל
    if frame.shape[0] > 32000 or frame.shape[1] > 32000:
        frame = cv2.resize(frame, (640, 480))
        print("Resized frame to 640x480 to avoid OpenCV errors.")

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
*****************************
ERROR: 
    >>> %Run object_detection.py
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1747351686.182597    3420 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.5.5) /tmp/bazel_build/opencv/modules/imgproc/src/imgwarp.cpp:1724: error: (-215:Assertion failed) dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX in function 'remap'


Process ended with exit code -6.

************************************************
    #ONLY FACE RECO TO TEST
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


   

  
  
