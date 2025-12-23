PROJECT: ANIME POSTER SUCCESS PREDICTION (VISION & HYBRID MODELS)

----------------------------------------
HOW TO RUN THE PROJECT (EXECUTION ORDER)
----------------------------------------

You should have 4 main folders before starting.
1. AnimeScorePredictionAllParameters
2. AnimeScorePredictionOnlyPoster
3. data
4. data/images

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


------------
REQUIREMENTS
------------
- Python 3.10
- TensorFlow / Keras (3.10.1 / 3.10.0)
- OpenCV (cv2)
- Pandas, NumPy
- Ultralytics (for YOLO)
- Tqdm (for progress bars)


Emre Özçatal
N25123078



