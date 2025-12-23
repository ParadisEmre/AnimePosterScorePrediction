# Anime Poster Success Prediction (Vision & Hybrid Models)

**Author:** Emre Özçatal

This project aims to predict the success score of anime series based on their promotional posters and metadata. It utilizes two distinct deep learning approaches: a **Hybrid Model** (Metadata + Images) and a **Vision-Only Model** (CNN). Additionally, the project includes Explainable AI (XAI) using Grad-CAM and Object Detection using YOLO to interpret the model's decisions.

---

## 📂 Project Structure

Ensure your local directory has the following structure before running the scripts. 
*(Note: Data and Model files are excluded from the repository by `.gitignore`)*.

```text
├── AnimeScorePredictionAllParameters/   # Scripts for the Hybrid Model
│   ├── animePosterScore.py
│   └── animePosterModelParameterWeightTest.py
├── AnimeScorePredictionOnlyPoster/      # Scripts for the Vision-Only Model
│   ├── animePosterScore.py
│   ├── animePosterAnalysis.py               # Grad-CAM Visualization
│   └── animePosterScoreObjectDetection.py   # YOLO Object Detection
├── data/                                # Data folder (created locally)
│   └── images/                          # Image downloads go here
└── README.md


KESİNLİKLE KOYMAN GEREKİYOR.

Şu anki haliyle sadece "Bu proje nedir ve dosyalar nerede?" sorusunu cevaplıyorsun. Ama bir yazılımcı (veya hocan) projeyi indirdiğinde "Bunu hangi sırayla çalıştıracağım?", "Hangi kütüphaneler lazım?" sorularının cevabını bulamazsa proje "eksik" görünür.

Önceki metindeki Execution Order (Çalıştırma Sırası) ve Requirements (Gereksinimler) kısımları hayati önem taşıyor.

Senin az önce verdiğin yeni klasör yapısına göre (dosyaları klasörlerin içine dağıtmışsın, bu daha düzenli olmuş) yolları güncelleyerek FİNAL ve TAM SÜRÜMÜ birleştirdim.

Bunu direkt kopyala yapıştır, mükemmel olacak:

Markdown

# Anime Poster Success Prediction (Vision & Hybrid Models)

**Author:** Emre Özçatal

This project aims to predict the success score of anime series based on their promotional posters and metadata. It utilizes two distinct deep learning approaches: a **Hybrid Model** (Metadata + Images) and a **Vision-Only Model** (CNN). Additionally, the project includes Explainable AI (XAI) using Grad-CAM and Object Detection using YOLO to interpret the model's decisions.

---

## 📂 Project Structure

Ensure your local directory has the following structure before running the scripts. 
*(Note: Data and Model files are excluded from the repository via `.gitignore`)*.

```text
├── AnimeScorePredictionAllParameters/   # Scripts for the Hybrid Model
│   ├── animePosterScore.py
│   └── animePosterModelParameterWeightTest.py
├── AnimeScorePredictionOnlyPoster/      # Scripts for the Vision-Only Model
│   ├── animePosterScore.py
│   ├── animePosterAnalysis.py               # Grad-CAM Visualization
│   └── animePosterScoreObjectDetection.py   # YOLO Object Detection
├── data/                                # Data folder (created locally)
│   └── images/                          # Image downloads go here
├── requirements.txt                     # Dependencies
└── README.md


⚙️ Requirements
Python: 3.10
Deep Learning: TensorFlow / Keras (3.10.1 / 3.10.0)
Computer Vision: OpenCV (cv2), Ultralytics (YOLO)
Data Processing: Pandas, NumPy
Utilities: Tqdm (for progress bars)

To install run: pip install tensorflow opencv-python pandas numpy ultralytics tqdm

----------------------------------------
🚀 HOW TO RUN THE PROJECT (EXECUTION ORDER)
----------------------------------------

STEP 1: HYBRID MODEL TRAINING (Metadata + Images)
Location: 
Folder: 'AnimeScorePredictionAllParameters'
File: animePosterScore.py

- Run this script first.
- It does the following:
  1. Downloads all anime posters to data/images folder (if downloaded already comment). --> IMPORTANT THIS DOWNLOADS ALL THE IMAGES FOR OTHER MODEL TOO
  2. Processes metadata --> 'ani_data.json' (JSON) and images --> 'ani_img.json' together into 'ani_data_merged.csv' in data folder.
  3. Trains the Hybrid Model and saves the weights 'anime_hybrid_model.h5'.


STEP 2: VISION-ONLY MODEL TRAINING 
Location: 
Folder: 'AnimeScorePredictionOnlyPoster'
File: animePosterScore.py

- Run this script after Step 1. --> IMPORTANT THIS SCRIPT MODEL IS USED IN GRADCAM AND YOLO
- It trains the model using only the images to create a pure CNN model using 'ani_data_merged.csv'.
- The end model is 'anime_vision_only_model.h5'


STEP 3: WEIGHT COMPARISON
File: animePosterScoreModelParameterWeightTest.py

- Run this script after Step 1 is trained to make sure everything is fine.
- It loads anime_hybrid_model.h5 and compares their features.


STEP 4: VISUALIZATION AND OBJECT DETECTION
A) Grad-CAM Visualization:
   File: animePosterAnalysis.py
   - This script uses the model trained in STEP 2 to generate heatmaps, showing where the model focuses on the poster.

B) YOLO Object Detection:
   File: animePosterScoreObjectDetection.py
   - Run the YOLO script to detect objects within the posters.
   
   (If the object counting process has already been performed and saved to 'anime_all_objects_detected.csv' file, comment the object counting function in the code to save time.)

📝 Notes
Data & Models: The data/ folder containing images and the trained .h5 model files are not included in this repository due to size constraints. Step 1 will handle the necessary data downloads.
Environment: It is recommended to run this project in a virtual environment (venv or conda) to avoid dependency conflicts.


