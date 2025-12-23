# 🎌 Anime Poster Community Score Prediction (Vision & Hybrid Models)

**Author:** [ParadisEmre](https://github.com/ParadisEmre)

This project aims to predict the community score of anime series using their posters and metadata.
Two different deep learning approaches are implemented:

* **🧠 Hybrid Model:** Combines metadata + poster images.
* **👁️ Vision-Only Model:** Uses only poster images (CNN).

In addition, the project includes Explainable AI (XAI) techniques using Grad-CAM and Object Detection using YOLO to interpret and visualize model decisions.

---

## 📂 Project Structure
Ensure your local directory has the following structure before running the scripts:

⚠️ **Note:** Data and trained models are excluded from the repository via .gitignore.

```text
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
├── .gitignore                           # Excludes heavy files
└── README.md
```

## ⚙️ Requirements

* Python v3.10.11
* TensorFlow v2.10.1 / Keras v2.10.0
* OpenCV (`cv2`) v4.12.0
* Ultralytics (YOLO) v8.3.235
* Pandas v2.3.3
* NumPy v1.23.5
* Tqdm v4.67.1

### 🔧 Installation

It is strongly recommended to use a virtual environment.

```bash
pip install tensorflow==2.10.1 opencv-python==4.12.0 ultralytics==8.3.235 pandas==2.3.3 numpy==1.23.5 tqdm==4.67.1
```

## 🚀 How to Run the Project (Execution Order)

Follow the steps in order to ensure all dependencies, data, and models are correctly generated.

### 🔹 STEP 1: Hybrid Model Training (Metadata + Images)
* **Folder:** `AnimeScorePredictionAllParameters`
* **File:** `animePosterScore.py`

**What this script does:**
1.  📥 **Downloads** all anime posters into `data/images` (⚠️ **IMPORTANT:** These images are reused by all other models).
2.  🔗 **Merges** Metadata (`ani_data.json`) and Image data (`ani_img.json`).
3.  🧪 **Creates** `ani_data_merged.csv` inside the `data/` folder.
4.  🧠 **Trains** the Hybrid Model and saves it as `anime_hybrid_model.h5`.

```bash
cd AnimeScorePredictionAllParameters
python animePosterScore.py
```

### 🔹 STEP 2: Vision-Only Model Training (CNN)
* **Folder:** `AnimeScorePredictionOnlyPoster`
* **File:** `animePosterScore.py`

**Notes:**
* ⚠️ Must be run **after STEP 1**.
* Uses `ani_data_merged.csv` (generated in the previous step).
* Trains a pure CNN model using only poster images.
* 📦 **Output:** `anime_vision_only_model.h5` (Required for Grad-CAM and YOLO steps).

```bash
cd ../AnimeScorePredictionOnlyPoster
python animePosterScore.py
```

### 🔹 STEP 3: Hybrid Model Weight Comparison
* **Folder:** `AnimeScorePredictionAllParameters`
* **File:** `animePosterModelParameterWeightTest.py`

**Purpose:**
* Loads `anime_hybrid_model.h5`.
* Analyzes and compares learned feature weights.
* Used for model validation and sanity checking.

```bash
cd ../AnimeScorePredictionAllParameters
python animePosterModelParameterWeightTest.py
```

### 🔹 STEP 4: Visualization & Object Detection

#### 🟠 A) Grad-CAM Visualization (Explainable AI)
* **Folder:** `AnimeScorePredictionOnlyPoster`
* **File:** `animePosterAnalysis.py`

**Description:**
* Uses the Vision-Only Model.
* Generates Grad-CAM heatmaps.
* Visualizes which regions of the poster influence predictions.

```bash
cd ../AnimeScorePredictionOnlyPoster
python animePosterAnalysis.py
```

#### 🟢 B) YOLO Object Detection
* **Folder:** `AnimeScorePredictionOnlyPoster`
* **File:** `animePosterScoreObjectDetection.py`

**Description:**
* Detects objects inside anime posters using YOLO.
* Counts and logs detected objects.

```bash
python animePosterScoreObjectDetection.py
```

## 📝 Notes

* **📁 Data & Models:** The `data/` folder and `.h5` model files are not included in this repository due to size limitations. **STEP 1** automatically handles all required downloads.
* **🧪 Environment:** It is recommended to use a virtual environment (`venv` or `conda`) to avoid dependency conflicts.