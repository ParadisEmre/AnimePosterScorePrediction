🎌 Anime Poster Success Prediction (Vision & Hybrid Models)

Author: ParadisEmre

This project aims to predict the success score of anime series using their promotional posters and metadata.
Two different deep learning approaches are implemented:

🧠 Hybrid Model — Combines metadata + poster images

👁️ Vision-Only Model — Uses only poster images (CNN)

In addition, the project includes Explainable AI (XAI) techniques using Grad-CAM and Object Detection using YOLO to interpret and visualize model decisions.

📂 Project Structure

Ensure your local directory has the following structure before running the scripts.
⚠️ Data and trained models are excluded from the repository via .gitignore.

├── AnimeScorePredictionAllParameters/   # Hybrid Model (Metadata + Images)
│   ├── animePosterScore.py
│   └── animePosterModelParameterWeightTest.py
│
├── AnimeScorePredictionOnlyPoster/      # Vision-Only Model (CNN)
│   ├── animePosterScore.py
│   ├── animePosterAnalysis.py               # Grad-CAM Visualization
│   └── animePosterScoreObjectDetection.py   # YOLO Object Detection
│
├── data/                                # Created locally
│   └── images/                          # Downloaded anime posters
│
└── README.md

⚙️ Requirements

Python 3.10
TensorFlow / Keras — 3.10.x
OpenCV (cv2)
Ultralytics (YOLO)
Pandas
NumPy
Tqdm

🔧 Installation

It is strongly recommended to use a virtual environment.
pip install tensorflow opencv-python pandas numpy ultralytics tqdm

🚀 How to Run the Project (Execution Order)

Follow the steps in order to ensure all dependencies, data, and models are correctly generated.
🔹 STEP 1: Hybrid Model Training (Metadata + Images)
📁 Folder: AnimeScorePredictionAllParameters
📄 File: animePosterScore.py

What this script does:
📥 Downloads all anime posters into data/images
⚠️ IMPORTANT:
These images are reused by all other models

🔗 Merges:
Metadata (ani_data.json)
Image data (ani_img.json)

🧪 Creates:
ani_data_merged.csv inside the data/ folder
🧠 Trains the Hybrid Model
💾 Saves the model as:
anime_hybrid_model.h5

🔹 STEP 2: Vision-Only Model Training (CNN)
📁 Folder: AnimeScorePredictionOnlyPoster
📄 File: animePosterScore.py

Notes:
⚠️ Must be run after STEP 1
Uses ani_data_merged.csv
Trains a pure CNN model using only poster images
This model is required for Grad-CAM and YOLO

📦 Output:
anime_vision_only_model.h5

🔹 STEP 3: Hybrid Model Weight Comparison
📁 Folder: AnimeScorePredictionAllParameters
📄 File:animePosterModelParameterWeightTest.py

Purpose:
Loads anime_hybrid_model.h5
Analyzes and compares learned feature weights
Used for model validation and sanity checking

🔹 STEP 4: Visualization & Object Detection
🟠 Grad-CAM Visualization (Explainable AI)

📁 Folder: AnimeScorePredictionOnlyPoster
📄 File:animePosterAnalysis.py

Uses the Vision-Only Model
Generates Grad-CAM heatmaps
Visualizes which regions of the poster influence predictions

🟢 YOLO Object Detection
📁 Folder: AnimeScorePredictionOnlyPoster
📄 File: animePosterScoreObjectDetection.py

Detects objects inside anime posters using YOLO
Counts and logs detected objects

📌 Optimization Tip:
If object detection results are already saved as:
anime_all_objects_detected.csv

You can comment out the object counting function in the script to save execution time.

📝 Notes
📁 Data & Models
The data/ folder and .h5 model files are not included due to size limitations.
STEP 1 automatically handles all required downloads.

🧪 Environment
Use venv or conda to avoid dependency conflicts.