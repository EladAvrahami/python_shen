# video analasys with gemini:
import cv2
import google.generativeai as genai
import imageio
import numpy as np
import time
import io
import pyttsx3
import os
from dotenv import load_dotenv

# --- Settings ---
# ===================================================================
#  הקטע הבא טוען את מפתח ה-API מהקובץ .env
#  ולא חושף אותו בקוד עצמו.
# ===================================================================
load_dotenv()  # טוען את המשתנים מהקובץ .env

API_KEY = os.getenv("API_KEY") # קורא את המפתח מהמשתנים שנטענו

# בדיקה אם המפתח נטען בהצלחה
if not API_KEY:
    print("Error: API_KEY not found.")
    print("Please make sure you have a .env file in the project directory")
    print("with the following content: API_KEY='your_actual_api_key'")
    exit()
else:
    # אם המפתח תקין, נסה להגדיר אותו
    try:
        genai.configure(api_key=API_KEY)
        print("Gemini API key configured successfully from .env file.")
    except Exception as e:
        print(f"An error occurred during API configuration: {e}")
        exit()
# ===================================================================

CAMERA_INDEX = 0  # 0 בדרך כלל למצלמת ברירת המחדל ב-Windows
CHUNK_DURATION_SECONDS = 5
FRAMES_PER_SECOND_CAPTURE = 15
MODEL_NAME = "gemini-1.5-flash-latest"

# --- Initialize Gemini Model ---
generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 30,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)


def capture_and_process_video_stream():
    # אתחול מנוע הדיבור
    try:
        # על ווינדוס, שורה זו תשתמש במנוע הדיבור המובנה SAPI5
        tts_engine = pyttsx3.init()
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        tts_engine = None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        error_msg = f"Error: Cannot open camera at index {CAMERA_INDEX}."
        print(error_msg)
        if tts_engine:
            tts_engine.say(error_msg)
            tts_engine.runAndWait()
        return

    print("Camera opened successfully. Press 'q' in the camera window to quit.")
    if tts_engine:
        tts_engine.say("Camera opened successfully.")
        tts_engine.runAndWait()

    previous_description_text = ""

    try:
        while True:
            frames_for_chunk = []
            start_time_chunk_capture = time.time()

            print(f"\nStarting frame collection for a {CHUNK_DURATION_SECONDS}-second chunk...")

            for i in range(int(FRAMES_PER_SECOND_CAPTURE * CHUNK_DURATION_SECONDS)):
                ret, frame = cap.read()
                if not ret:
                    print("Error capturing frame.")
                    time.sleep(0.1)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_for_chunk.append(frame_rgb)

                cv2.imshow("Live Camera Feed (press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                time.sleep(1 / FRAMES_PER_SECOND_CAPTURE)

            if not frames_for_chunk:
                print("No frames were captured for this chunk. Skipping API call.")
                continue

            print("Encoding video chunk...")
            video_bytes_io = io.BytesIO()
            imageio.mimwrite(video_bytes_io, frames_for_chunk, format='mp4', fps=FRAMES_PER_SECOND_CAPTURE, quality=7)
            video_bytes = video_bytes_io.getvalue()

            if not video_bytes:
                print("Error: Video data is empty after encoding.")
                continue

            print(f"Sending to Gemini for analysis...")

            contents_for_api = [
                {'mime_type': 'video/mp4', 'data': video_bytes},
                "Based on the video segment, describe the current scene and ongoing actions in English."
            ]
            if previous_description_text:
                contents_for_api.insert(0, f"Context from previous analysis: {previous_description_text}")

            message_to_speak = ""
            try:
                response = model.generate_content(contents_for_api, request_options={"timeout": 120})

                if response.parts:
                    message_to_speak = response.text.strip()
                    print(f"\n--- Gemini's Description ---")
                    print(message_to_speak)
                    print("--------------------------")
                else:
                    message_to_speak = "No specific description was generated."
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        message_to_speak = f"Content blocked due to: {response.prompt_feedback.block_reason.name}."
                    print(message_to_speak)

                if tts_engine and message_to_speak:
                    tts_engine.say(message_to_speak)
                    tts_engine.runAndWait()

                previous_description_text = message_to_speak if response.parts else ""

            except Exception as e:
                error_message = f"Error communicating with Gemini API: {e}"
                print(error_message)
                if tts_engine:
                    tts_engine.say("API communication error.")
                    tts_engine.runAndWait()

    except KeyboardInterrupt:
        print("\nUser requested to exit.")
        if tts_engine:
            tts_engine.say("Exiting program.")
            tts_engine.runAndWait()
    finally:
        print("Closing camera and resources...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_process_video_stream()