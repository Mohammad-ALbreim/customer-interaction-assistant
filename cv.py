import tkinter as tk
from tkinter import ttk, font
import cv2
from PIL import Image, ImageTk
import threading
import time
import queue
import os
import google.generativeai as genai
from ultralytics import YOLO
from dotenv import load_dotenv

import json

load_dotenv()

# --- Configuration ---
# Load the YOLO model (yolov8n.pt is a small, fast model)
YOLO_MODEL = YOLO('yolov8n.pt')

# Configure the Gemini API with the key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Missing GEMINI_API_KEY. Copy .env.example to .env and add your key.")
    exit()

try:
    genai.configure(api_key=API_KEY)
    GEMINI_MODEL = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit()

# Enhanced prompt for more robust emotion detection
GEMINI_PROMPT = """
You are an expert in non-verbal communication, micro-expressions, and customer service psychology. Analyze the provided image of a person with extreme attention to detail.

Focus on these specific indicators:
- Facial expressions: eyebrows, eyes, mouth, cheek tension
- Body language: posture, shoulder position, hand gestures
- Overall demeanor: energy level, attention focus, stress indicators
- Cultural context: respect cultural differences in expression

Provide your analysis in clean JSON format with these keys:
1. "emotional_state": A precise, descriptive string combining primary and secondary emotions (e.g., 'Anxious but Hopeful', 'Confident and Engaged', 'Tired but Patient', 'Frustrated but Controlled')
2. "confidence_score": A number from 0.0 to 1.0 indicating how confident you are in your assessment
3. "key_indicators": A brief list of the main visual cues you observed (e.g., ['furrowed brow', 'tight smile', 'forward lean'])
4. "service_recommendation": Detailed, actionable advice for service workers (2-3 specific strategies)
5. "interaction_style": Recommended approach ('gentle', 'efficient', 'enthusiastic', 'patient', 'professional')

If no clear person is visible, return all values as "N/A" except confidence_score (set to 0.0).
Be precise, professional, and avoid definitive judgments about character or intentions.
"""

class EmotionAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Interaction Assistant")
          # --- State Variables ---
        self.last_api_call_time = 0
        self.api_cooldown = 10  # seconds between API calls
        self.api_thread = None
        self.result_queue = queue.Queue()
        self.person_detected_in_frame = False
        self.current_emotion = "--"
        self.current_recommendation = "--"

        # --- Configure Styles and Fonts ---
        self.root.configure(bg="#2E2E2E")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2E2E2E')
        self.style.configure('TLabel', background='#2E2E2E', foreground='white')
        self.title_font = font.Font(family="Segoe UI", size=14, weight="bold")
        self.label_font = font.Font(family="Segoe UI", size=11, weight="bold")
        self.text_font = font.Font(family="Segoe UI", size=10)
        self.status_font = font.Font(family="Segoe UI", size=9, slant="italic")        # --- Create GUI Layout ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1) # Full width for video feed
        main_frame.rowconfigure(0, weight=1)

        # Video Feed - Full width
        video_frame = ttk.Frame(main_frame, padding=5)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Initializing camera...")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5, font=self.status_font, foreground="#A9A9A9")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Initialize Video Capture ---
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            self.status_var.set("Error: Cannot open camera.")
            return

        self.update_frame()

    def perform_yolo_detection(self, frame):
        """Runs YOLO model on the frame and returns the processed frame and cropped person image."""
        results = YOLO_MODEL(frame, verbose=False)
        person_crop = None
        self.person_detected_in_frame = False
        
        for r in results:
            for box in r.boxes:
                # Check if the detected object is a person (class 0 in COCO dataset)
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Draw a green bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Crop the original frame to get the person's image
                    person_crop = frame[y1:y2, x1:x2]
                    self.person_detected_in_frame = True
                    break # Process only the first person found
            if self.person_detected_in_frame:
                break
        
        return frame, person_crop

    def draw_text_overlay(self, frame, text, position, font_scale=0.8, color=(255, 255, 255), thickness=2, bg_color=None):
        """Draw text overlay on the frame with optional background."""
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Calculate position
        x, y = position
        
        # Draw background rectangle if specified
        if bg_color:
            padding = 10
            cv2.rectangle(frame, 
                         (x - padding, y - text_height - padding), 
                         (x + text_width + padding, y + baseline + padding), 
                         bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return text_height + baseline + 20  # Return height for next line positioning

    def add_overlays_to_frame(self, frame):
        """Add emotional state and recommendation overlays to the frame."""
        frame_height, frame_width = frame.shape[:2]
        
        # Position for overlays (top-left area)
        start_x = 20
        start_y = 50
        
        # Draw emotional state
        if self.current_emotion != "--":
            emotion_text = f"Emotional State: {self.current_emotion}"
            line_height = self.draw_text_overlay(frame, emotion_text, (start_x, start_y), 
                                               font_scale=0.7, color=(100, 255, 100), thickness=2, 
                                               bg_color=(0, 0, 0))
            start_y += line_height
        
        # Draw recommendation (split into multiple lines if too long)
        if self.current_recommendation != "--":
            recommendation_text = f"Recommendation: {self.current_recommendation}"
            max_chars_per_line = 60  # Adjust based on your needs
            
            # Split long text into multiple lines
            words = recommendation_text.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line += (" " + word) if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw each line
            for line in lines:
                line_height = self.draw_text_overlay(frame, line, (start_x, start_y), 
                                                   font_scale=0.6, color=(100, 200, 255), thickness=2, 
                                                   bg_color=(0, 0, 0))
                start_y += line_height
        
        return frame

    def get_gemini_analysis(self, image_crop, result_queue):
        """Sends the cropped image to Gemini API and puts the result in a queue."""
        try:
            self.status_var.set("Analyzing emotion with Gemini...")
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            img_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            response = GEMINI_MODEL.generate_content([GEMINI_PROMPT, pil_image])
            
            # Clean up the response to extract the JSON part
            text_response = response.text.strip().replace("```json", "").replace("```", "")
            analysis_result = json.loads(text_response)
            
            result_queue.put(analysis_result)
        except Exception as e:
            print(f"Gemini API Error: {e}")
            result_queue.put({"error": "Failed to get analysis."})

    def update_frame(self):
        """The main loop of the application."""
        ret, frame = self.vid.read()
        if not ret:
            self.status_var.set("Error: Can't receive frame. Exiting.")
            self.root.after(1000, self.on_closing)
            return

        # Perform YOLO detection
        processed_frame, person_crop = self.perform_yolo_detection(frame)

        # Trigger API call logic
        current_time = time.time()
        is_api_ready = (current_time - self.last_api_call_time > self.api_cooldown)
        is_thread_inactive = (self.api_thread is None or not self.api_thread.is_alive())

        if self.person_detected_in_frame and is_api_ready and is_thread_inactive:
            self.last_api_call_time = current_time
            self.api_thread = threading.Thread(target=self.get_gemini_analysis, args=(person_crop, self.result_queue))
            self.api_thread.start()
        elif not self.person_detected_in_frame and is_thread_inactive:
            self.status_var.set("Scanning for customers...")        # Check for results from the Gemini thread
        try:
            result = self.result_queue.get_nowait()
            if "error" in result:
                self.current_emotion = "Error"
                self.current_recommendation = result["error"]
                self.status_var.set("An error occurred during analysis.")
            else:
                self.current_emotion = result.get("emotional_state", "N/A")
                self.current_recommendation = result.get("service_recommendation", "N/A")
                self.status_var.set("Analysis complete. Standing by.")
        except queue.Empty:
            pass # No new result yet

        # Add overlays to the frame
        processed_frame = self.add_overlays_to_frame(processed_frame)

        # Update the video feed in the GUI
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # Resize image to fit the label, maintaining aspect ratio (make it bigger)
        img.thumbnail((1200, 900))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule the next frame update
        self.root.after(20, self.update_frame)

    def on_closing(self):
        """Release resources when the window is closed."""
        print("Closing application...")
        self.vid.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()