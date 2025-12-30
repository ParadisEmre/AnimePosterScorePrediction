import pandas as pd
import numpy as np
import tensorflow as tf
import os
import ast
import re
from tqdm import tqdm
import time
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import gc

# GPU growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

MODEL_PATH = 'anime_hybrid_model.h5'
MERGED_CSV_PATH = '../data/ani_data_merged.csv' 
DOWNLOADED_IMG_PATH = '../data/images/'

def clean_episodes(ep_str):
    try:
        return float(ep_str)
    except:
        return np.nan

def load_and_preprocess_data():
    
    data = pd.read_csv(MERGED_CSV_PATH, low_memory=False)
    
    # Data clean up (JSON Data Check)
    data.dropna(subset=['score'], inplace=True)
    data['score'] = pd.to_numeric(data['score'], errors='coerce')
    data.dropna(subset=['score'], inplace=True)
    data.dropna(subset=['genres'], inplace=True)

    # total_episodes clean up
    data['episodes_count'] = data['total_episodes'].apply(clean_episodes)
    data.dropna(subset=['episodes_count'], inplace=True)
    # popularity clean up
    data['popularity_val'] = pd.to_numeric(data['popularity'], errors='coerce').fillna(0)
    data.dropna(subset=['popularity_val'], inplace=True)

    # Normalization to value between 0 1 for each rating type
    data = pd.get_dummies(data, columns=['rating'], dummy_na=False)

    # Normalization to value between 0 1 for each genre type
    data['genres_list'] = data['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    all_genres = sorted(list(set(x for l in data['genres_list'] for x in l if x)))
    for genre in all_genres:
        data[f"Genre_{genre}"] = data['genres_list'].apply(lambda x: 1 if genre in x else 0)

    # Normalization to value between 0 1
    cols_to_normalize = ['episodes_count', 'popularity_val']
    
    for col in cols_to_normalize:
        min_val = data[col].min()
        max_val = data[col].max()
        if max_val > min_val:
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0
    
    # Reset index
    data = data.reset_index(drop=True)
    
    # Original index is preserved in file_id without the shifts
    data['file_id'] = data.index
    
    # Data clean up (Image Data Check)
    print("Check Images")
    
    if os.path.exists(DOWNLOADED_IMG_PATH):
        existing_files = set(os.listdir(DOWNLOADED_IMG_PATH))
        # Keep only image consisting image
        data = data[data['file_id'].apply(lambda x: f"{x}.jpg" in existing_files)].copy()
        print(f"Cleaned data count --> {len(data)}")
    print("Check Images Done")
    # Check can not catch up
    time.sleep(3)    
    
    # The columns
    tab_cols = cols_to_normalize + \
               [c for c in data.columns if c.startswith('rating_')] + \
               [c for c in data.columns if c.startswith('Genre_')]
    
    # To make sure
    data = data.reset_index(drop=True)
    
    return data, tab_cols


def analyze_weights_of_fields_importance():

    # Data and Features Selected
    data, tab_cols = load_and_preprocess_data()

    # Model
    model = load_model(MODEL_PATH) 

    test_data = data.sample(n=500, random_state=42).reset_index(drop=True)
    
    # Images
    X_images = np.zeros((len(test_data), 224, 224, 3), dtype=np.float16)
    for i, row in tqdm(test_data.iterrows(), total=len(test_data)):
        img_path = os.path.join(DOWNLOADED_IMG_PATH, f"{row['file_id']}.jpg")
        # Img loading
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            X_images[i] = preprocess_input(img).astype(np.float16)
        else:
            X_images[i] = np.zeros((224, 224, 3), dtype=np.float16) # If fail

    # Table
    X_tab = test_data[tab_cols].values.astype(np.float32)
    y_true = test_data['score'].values

    # Base Prediction
    base_preds = model.predict([X_images, X_tab], verbose=0, batch_size=8)
    base_mae = np.mean(np.abs(base_preds.flatten() - y_true))
    print(f"\nBase Prediction MAE: {base_mae:.4f}")

    results = []
    features_to_test = ['Poster'] + tab_cols 

    for feature in tqdm(features_to_test):
        
        # Keep table or img data still, shuffle other
        if feature == 'Poster':
            X_img_shuffled = X_images.copy()
            np.random.shuffle(X_img_shuffled)
            X_tab_shuffled = X_tab
            
        else:
            X_img_shuffled = X_images 
            X_tab_shuffled = X_tab.copy()
            col_idx = tab_cols.index(feature) 
            np.random.shuffle(X_tab_shuffled[:, col_idx])
            
        # New Predictions with mix
        new_preds = model.predict([X_img_shuffled, X_tab_shuffled], verbose=0, batch_size=8)
        new_mae = np.mean(np.abs(new_preds.flatten() - y_true))
        
        importance = new_mae - base_mae
        results.append({'Feature': feature, 'Importance': importance})

        # To clear memory
        if feature == 'Poster':
            del X_img_shuffled
        else:
            del X_tab_shuffled
        del new_preds
        # To clear memory
        gc.collect() 

    # Res
    res_data = pd.DataFrame(results).sort_values(by='Importance', ascending=False)
    print(res_data)

if __name__ == "__main__":
    analyze_weights_of_fields_importance()