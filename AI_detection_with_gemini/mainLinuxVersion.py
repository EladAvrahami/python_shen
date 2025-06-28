# video analasys with gemini On linux:
# Replace with my actual Gemini API key
# GOOGLE_API_KEY = "AIKEY"
import cv2
import google.generativeai as genai
# REMOVED: from google.generativeai.types import Part
import imageio
import numpy as np
import os
import time
import io  # For working with in-memory byte streams

# --- Settings ---
try:
    genai.configure(api_key=os.environ["AIKEY"])
except KeyError:
    GOOGLE_API_KEY_FALLBACK = "AIKEY"  # <<<<<<< ENTER YOUR API KEY HERE IF IT'S NOT AN ENVIRONMENT VARIABLE
    if GOOGLE_API_KEY_FALLBACK:
        genai.configure(api_key=GOOGLE_API_KEY_FALLBACK)
        print("Using API key provided directly in the code.")
    else:
        print(
            "Error: The GOOGLE_API_KEY environment variable is not set, and no API key was provided directly in the code.")
        print("Please set it or enter the API key in the GOOGLE_API_KEY_FALLBACK variable.")
        exit()

CAMERA_INDEX = 0
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
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {CAMERA_INDEX}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_capture_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera opened: {frame_width}x{frame_height} @ ~{actual_capture_fps:.2f} FPS (attempting to capture at {FRAMES_PER_SECOND_CAPTURE} FPS)")

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

                current_frame_time_target = (i + 1) / FRAMES_PER_SECOND_CAPTURE
                elapsed_in_frame_loop = time.time() - start_time_chunk_capture
                sleep_duration = max(0, current_frame_time_target - elapsed_in_frame_loop)
                time.sleep(sleep_duration)

            if not frames_for_chunk:
                print("No frames were captured for this chunk. Skipping API call.")
                continue

            print(f"Captured {len(frames_for_chunk)} frames. Creating in-memory video chunk...")

            video_bytes_io = io.BytesIO()
            try:
                imageio.mimwrite(video_bytes_io, frames_for_chunk, format='mp4', fps=FRAMES_PER_SECOND_CAPTURE,
                                 quality=7)
            except Exception as e:
                print(f"Error creating video with imageio: {e}")
                print("Ensure ffmpeg is installed and accessible: pip install imageio-ffmpeg")
                continue

            video_bytes = video_bytes_io.getvalue()

            if not video_bytes:
                print("Error: Video data is empty after encoding. Skipping API call.")
                continue

            print(f"Video chunk created (size: {len(video_bytes) / 1024:.2f} KB). Sending to Gemini...")

            # Constructing the content list without explicit Part objects
            contents_for_api = []
            if previous_description_text:
                # Add previous context as a simple string
                contents_for_api.append(f"Context from previous analysis: {previous_description_text}")

            # Add video data as a dictionary
            contents_for_api.append({'mime_type': 'video/mp4', 'data': video_bytes})

            # Add current prompt as a simple string
            current_instruction_prompt = "Based on the video segment and any previous context provided, describe the current scene and ongoing actions."
            if not previous_description_text:
                current_instruction_prompt = "Describe the scene and ongoing actions in this initial video segment."
            contents_for_api.append(current_instruction_prompt)

            try:
                request_options = {"timeout": 120}
                response = model.generate_content(contents_for_api, request_options=request_options)

                current_description = ""  # Initialize
                if response.parts:
                    current_description = response.text
                elif response.candidates and response.candidates[0].finish_reason != 'STOP':
                    print(f"Content generation stopped. Reason: {response.candidates[0].finish_reason}")
                    if response.prompt_feedback:
                        print(f"Prompt Feedback: {response.prompt_feedback}")
                else:
                    print("The API response was unusual or content might have been blocked.")
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        print(f"Prompt Feedback: {response.prompt_feedback}")

                if current_description:
                    print(f"\n--- Description from Gemini ({time.strftime('%H:%M:%S')}) ---")
                    print(current_description)
                    print("--------------------------------------")
                    previous_description_text = current_description
                else:
                    # If no description was generated (e.g. due to blocking or error)
                    # Don't carry over an empty description as context, or reset context
                    if previous_description_text:  # only print if there was a previous description
                        print("No new description generated for this chunk.")
                    previous_description_text = ""  # Reset context if blocked or no text


            except Exception as e:
                print(f"Error communicating with Gemini API: {e}")

    except KeyboardInterrupt:
        print("\nUser requested to exit.")
    finally:
        print("Closing camera and resources...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_process_video_stream()
