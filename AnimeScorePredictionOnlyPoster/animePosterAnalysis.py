import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input 

MODEL_PATH = 'anime_vision_only_model.h5'
MERGED_CSV_PATH = '../data/ani_data_merged.csv' 
DOWNLOADED_IMG_PATH = '../data/images/'
LAST_CONV_LAYER = 'conv5_block3_out' 

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Model
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        score_prediction = preds[:, 0] 

    grads = tape.gradient(score_prediction, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(heatmap, 0.6, img, 0.4, 0)
    return superimposed_img

def analyze_visuals():
    # Model
    model = load_model(MODEL_PATH)
    
    # Data
    data = pd.read_csv(MERGED_CSV_PATH, low_memory=False)
    
    # Original index is preserved in file_id without the shifts
    data['file_id'] = data.index
    
    
    print("Check Images")
    if os.path.exists(DOWNLOADED_IMG_PATH):
        existing_files = set(os.listdir(DOWNLOADED_IMG_PATH))
        
        # Keep only image consisting datas
        data = data[data['file_id'].apply(lambda x: f"{x}.jpg" in existing_files)].copy()
        print(f"Cleaned data count --> {len(data)}")
    print("Check Images Done")
    # Check can not catch up
    time.sleep(3)    
    
    
    # Delete empty scores and convert them to numbers
    data.dropna(subset=['score'], inplace=True)
    data['score'] = pd.to_numeric(data['score'], errors='coerce')
    # Some are broken after conversion so drop
    data.dropna(subset=['score'], inplace=True)
    
    # Animes to display
    highest_5_anime = data.sort_values(by='score', ascending=False).head(5)
    lowest_5_anime = data.sort_values(by='score', ascending=True).head(5)
    
    anime_list = pd.concat([highest_5_anime, lowest_5_anime])


    cols = 5
    rows = 2
    plt.figure(figsize=(15, 8)) 
    plot_idx = 1
    
    for i, row in anime_list.iterrows():
        try:
            # Img formatting
            img_path = os.path.join(DOWNLOADED_IMG_PATH, f"{row['file_id']}.jpg")

            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_resized = cv2.resize(img, (224, 224))
            img_array = preprocess_input(img_resized.astype(np.float32))
            img_batch = np.expand_dims(img_array, axis=0)

            # Prediction
            preds = model.predict(img_batch, verbose=0)
            score = preds[0][0]

            # Grad-CAM
            display_img = img
            try:
                heatmap = make_gradcam_heatmap(img_batch, model, LAST_CONV_LAYER)
                res = save_and_display_gradcam(img_path, heatmap)
                if res is not None: display_img = res
            except: pass

            # Plotting (4x10 Grid)
            ax = plt.subplot(rows, cols, plot_idx)
            plt.imshow(display_img)
            
            # İsim çok uzunsa kısalt (Görsel bozulmasın diye)
            anime_name = row['name']
            if len(anime_name) > 15:
                anime_name = anime_name[:12] + "..."
            
            # R: Real Score, P: Predicted Score
            plt.title(f"{anime_name}\nR:{row['score']:.1f}|P:{score:.1f}", fontsize=7)
            plt.axis("off")
            plot_idx += 1
            
        except Exception as e:
            print(f"Skipped an image due to error: {e}")

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    analyze_visuals()