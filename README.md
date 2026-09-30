# Customer Interaction Assistant

**A real-time computer vision assistant that reads a customer's emotional state from a live camera feed and suggests how a service worker should respond.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![Gemini](https://img.shields.io/badge/Google-Gemini_2.0_Flash-4285F4?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Overview

The app watches a camera feed, detects a person with YOLOv8, crops that person from the frame, and sends the crop to Google Gemini for an emotional read. The result is drawn back onto the live video: the detected emotional state, and a concrete recommendation for how to serve this customer.

It is built for a service desk, a reception, or a retail counter, where reading the customer quickly changes how you should talk to them.

## How it works

```
Camera frame
   -> YOLOv8 detects a person (COCO class 0) and draws a bounding box
   -> the person is cropped out of the frame
   -> the crop is sent to Gemini 2.0 Flash with a structured prompt
   -> Gemini returns JSON: emotional state, confidence, visual cues, recommendation
   -> the result is drawn as an overlay on the live video
```

Two design decisions keep the app responsive:

- **Non-blocking architecture.** The Gemini call runs in a separate thread and returns through a queue, so the video never freezes while the model is thinking.
- **API cooldown.** A 10-second cooldown between calls keeps cost and rate limits under control instead of sending one request per frame.

## What Gemini returns

The prompt asks for clean JSON with five fields:

| Field | Meaning |
|---|---|
| `emotional_state` | Primary and secondary emotion, e.g. "Anxious but Hopeful" |
| `confidence_score` | 0.0 to 1.0 |
| `key_indicators` | The visual cues behind the reading |
| `service_recommendation` | 2-3 concrete strategies for the service worker |
| `interaction_style` | gentle, efficient, enthusiastic, patient or professional |

## Setup

```bash
pip install -r requirements.txt
```

Download the YOLOv8 nano weights (not stored in this repository):

```bash
yolo export model=yolov8n.pt
```

Or let Ultralytics fetch them on first run.

Add your Gemini key:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

## Run

```bash
python cv.py
```

A window opens with the live feed. Green boxes mark detected people; the emotional reading and the recommendation appear as overlays.

## Requirements

- Python 3.10 or newer
- A working webcam
- A Google Gemini API key

## Notes and limits

- Emotion reading from a single image is an estimate, not a diagnosis. The prompt deliberately avoids judgments about character or intent.
- The app analyzes the first person detected in the frame.
- Camera index is fixed to `0` in `cv.py`; change it if you use a second camera.

## Tech stack

Python, YOLOv8 (Ultralytics), Google Gemini 2.0 Flash, OpenCV, Tkinter, Pillow, threading

---

Built by [Mohammad Albreim](https://github.com/Mohammad-ALbreim) as part of AI and Data Science work at the University of Petra.
