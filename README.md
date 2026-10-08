# OMI

**OMI (Omni Multimodal Interface)** is a Python-based desktop interface that combines computer vision, hand gestures, face recognition, and AI-powered face analysis.

## Features

- Real-time hand tracking
- Hand gesture detection
- Pinch detection
- Finger tracking
- Gesture-based mouse control
- Gesture-based scrolling
- Face detection
- Facial landmark tracking
- Face recognition using ArcFace
- Face age and gender estimation
- Face registration with multiple samples
- Profile card generation
- Real-time HUD and visual effects

## Project Structure

OMI/
│
├── main.py
├── face_system.py
├── test.py
├── modules/
│   ├── age_estimator.py
│   ├── effects.py
│   ├── face_age_clientscan.py
│   ├── face_database.py
│   ├── face_detector.py
│   ├── face_embedding_database.py
│   ├── face_embedding_extractor.py
│   ├── face_embedding_recognizer.py
│   ├── face_feature_extractor.py
│   ├── face_landmarker.py
│   ├── face_recognizer.py
│   ├── finger_state.py
│   ├── finger_tracker.py
│   ├── fps_counter.py
│   ├── gesture_detector.py
│   ├── hand_detector.py
│   ├── hand_stabilizer.py
│   ├── hud.py
│   ├── landmark_filter.py
│   ├── mouse_buttons.py
│   ├── mouse_controller.py
│   ├── pinch_detector.py
│   ├── profile_card.py
│   ├── renderer.py
│   ├── scroll_controller.py
│   ├── sound_manager.py
│   ├── tracker.py
│   └── webcam.py
│
└── models/
    └── Download required models separately

There are two main files: main.py and faces_system.py each having its own use
main: This file is for generic face detection and complex hand and hand gesture recognition. Also has the movable mouse feature which can be enabled through "c". Through this the user can move the cursor without the need of a mouse using their fingers. Common actions it can do is:
left click
right click
scroll up/down

face_system: This is the more complex file which is used for facial recognition and saving face data in a database. It's features include:
Face registration
Face detection
Face recognition
Age estimation
Gender recognition
A secured databases containing image samples of the user/subject
A OMI profile card for every person's detail stored in the database

NOTE: Both of these modules require certain requirement models listed below at ## Models

## Requirements

OMI requires:

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- ONNX Runtime
- Additional Python packages used by the modules

A compatible webcam is also required.

## Models

The model files are **not included in this repository** because some of them are very large.
Create a folder named:
models/

and place the required model files inside it.

Required models include:
age_googlenet.onnx
blaze_face_short_range.tflite
face_landmarker.task
faceage.onnx
fairface.onnx
fastface_age.onnx
hand_landmarker.task

Download each model from its original/official source and place it in the `models` folder using the exact filenames above.

## Installation

Clone the repository:
git clone https://github.com/TCPLAYD67/OMI.git
cd OMI

Install the required Python dependencies:

pip install -r requirements.txt

Place the required model files in:
models/

Then run the appropriate Python entry point.

## Controls

Common controls include:

- C     Toggle Cursor Control
- E     Show Debug Info
- R     Register a face
- L     Toggle facial landmark visualization
- F     Toggle fullscreen
- ESC   Cancel registration
- Q     Quit

Gesture controls are handled through the computer-vision system.

## Disclaimer

OMI is a computer-vision project intended for experimentation and educational use.

Recognition and AI-based predictions may not always be accurate.
