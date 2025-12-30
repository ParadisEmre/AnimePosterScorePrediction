import pandas as pd
import requests
import os
import time
import ast
import re
from tqdm import tqdm 
import numpy as np
import cv2 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Flatten, Dropout, Input, Concatenate, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision

MODEL_PATH = 'anime_hybrid_model.h5'
MERGED_CSV_PATH = '../data/ani_data_merged.csv'
IMG_JSON_PATH = '../data/ani_img.json'
DATA_JSON_PATH = '../data/ani_data.json'
DOWNLOADED_IMG_PATH = '../data/images/'
EPOCHS = 100 
BATCH_SIZE = 32 
TARGET_COLS = ['score']
TARGET_SIZE = (224, 224, 3)

mixed_precision.set_global_policy('mixed_float16') #RTX3060 prop
class AnimeDataGenerator(Sequence):

    # Constructor
    def __init__(self, data, img_dir, tab_cols, target_cols, batch_size, target_size, shuffle): 
        
        # To not alter the original data
        self.data = data.copy()
        
        # To not get error on indexes after deleting the animes with no poster
        self.data = self.data.reset_index(drop=True) 

        self.img_dir = img_dir # Poster path
        self.tab_cols = tab_cols # Columns that will be used in training
        self.batch_size = batch_size # How many images in each iteration
        self.target_size = target_size # Target dimension
        self.shuffle = shuffle # Shuffle
        self.target_cols = target_cols # The target columns to decide
        self.labels = data[target_cols].values # Target data as numpy array
        self.on_epoch_end()

    # How many times the data will be retrieved
    def __len__(self): 
        return int(np.floor(len(self.data) / self.batch_size)) 

    # On epoch end
    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True) # Sample is taken with random retrieve. Then the indexes are resetted
            self.labels = self.data[self.target_cols].values # Target data as numpy array

    # Index element retrieve
    def __getitem__(self, index):
        indexes = self.data.index[index * self.batch_size:(index + 1) * self.batch_size] # Retrieve index according to batch size
        X, y = self.__data_generation(indexes) # Data generation
        return X, y

    def __data_generation(self, indexes):
        X_img = np.empty((self.batch_size, *self.target_size), dtype=np.float32) # (32, 224, 224, 3) -> 32 img --> each 224x224 and RGB
        y = np.empty((self.batch_size, len(self.target_cols)), dtype=np.float32) # Result scores
        
        for i, row_idx in enumerate(indexes):
            file_index = self.data.at[row_idx, 'file_id'] # The unshuffled ids
            img_path = os.path.join(self.img_dir, f"{file_index}.jpg") # Img path
            
            img = cv2.imread(img_path) # Harddisk to RAM OpenCV

            if img is not None: # If img is downloaded successfully
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Color fix between ResNet and OpenCV
                img_resized = cv2.resize(img, self.target_size[:2]) # Resize
                X_img[i,] = resnet_preprocess(img_resized.astype(np.float32)) # Preprocessing - Normalization
                y[i,] = self.labels[row_idx] # Labeling
            else:
                # If no image, zero matrix
                 X_img[i,] = np.zeros(self.target_size)
                 y[i,] = self.labels[row_idx]
        
        # Metadata addition
        X_tab = self.data.iloc[indexes][self.tab_cols].values.astype(np.float32)

        return (X_img, X_tab), y
    

def build_hybrid_model(json_features):
    # Model
    input_img = Input(shape=TARGET_SIZE)
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_img)
    base_model.trainable = False 
    
    for layer in base_model.layers[-30:]: # Last 30 layers opened for training
        layer.trainable = True
    
    x1 = base_model.output
    x1 = GlobalAveragePooling2D()(x1) # 7x7x2048 to 1x1x2048 dont need the position info
    
    x1 = Dense(1024, kernel_regularizer=l2(0.01))(x1)
    x1 = BatchNormalization()(x1)       # Stabilize
    x1 = Activation('relu')(x1)         # Activation func
    x1 = Dropout(0.2)(x1)               # To stop memorizing (overfitting) butkeep its weight
    
    x1 = Dense(512, kernel_regularizer=l2(0.01))(x1)
    x1 = BatchNormalization()(x1)       # Stabilize
    x1 = Activation('relu')(x1)         # Activation func
    x1 = Dropout(0.15)(x1)               # To stop memorizing (overfitting) butkeep its weight
    
    x1 = Dense(256, kernel_regularizer=l2(0.01))(x1)
    x1 = BatchNormalization()(x1)       # Stabilize
    x1 = Activation('relu')(x1)         # Activation func
    x1 = Dropout(0.1)(x1)               # To stop memorizing (overfitting) butkeep its weight

    # JSON Data (popularity etc)
    input_tab = Input(shape=(json_features,))
    
    x2 = Dense(64, kernel_regularizer=l2(0.01))(input_tab) # To stop overfitting JSON PARAMS COOUNT 33 X2 NEURONS
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.3)(x2)
    
    # MLP
    x2 = Dense(32, kernel_regularizer=l2(0.01))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.3)(x2) # Lower dropout the data is important
    
    # Concat
    combined = Concatenate()([x1, x2])
    
    z = Dense(32, kernel_regularizer=l2(0.01))(combined)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Dropout(0.2)(z)
    
    output = Dense(1, activation='linear')(z)

    # Keep score between 0-10
    output = Lambda(lambda z: tf.clip_by_value(z, clip_value_min=0.0, clip_value_max=10.0))(output)

    model = Model(inputs=[input_img, input_tab], outputs=output)
    
    optimizer = Adam(learning_rate=0.0002) # Lowered to not break how ResNet50 is
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def download_all_images(data, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    downloaded_count = 0
    skipped_count = 0
    
    data_to_download = data.copy()
    data_to_download['file_id_temp'] = data_to_download.index 

    for index, row in tqdm(data_to_download.iterrows(), total=len(data_to_download)):
        img_url = row['img']
        file_id = row['file_id_temp']
        file_path = os.path.join(save_folder, f"{file_id}.jpg") 
        
        # To not download again
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            skipped_count += 1
            continue
        
        # What if the img is broken or not downloaded successfully
        
        try:
            # Connection timeout check
            response = requests.get(img_url, timeout=20, stream=True) 
            content_type = response.headers.get('Content-Type', '').lower()

            if response.status_code == 200 and 'image' in content_type:
                
                # Save temporary and check if its broken
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                
                # Try to read
                img = cv2.imread(file_path)
                
                # Delete broken image
                if img is None:
                    os.remove(file_path)
                    continue
                    
                downloaded_count += 1
                
            time.sleep(0.02)
            
        except requests.exceptions.RequestException:
            pass 
        except Exception as e:
            if os.path.exists(file_path):
                 os.remove(file_path)
            pass
            
    print(f"\n{downloaded_count} --> newly downloaded {skipped_count} --> skipped")


def clean_episodes(ep_str):
    try:
        return float(ep_str)
    except:
        return np.nan


if __name__ == "__main__":
    
    # Data Load
    if not os.path.exists(MERGED_CSV_PATH):
        # Convert to CSV data merging two
        json_ani_img_data = pd.read_json(IMG_JSON_PATH) 
        json_ani_data = pd.read_json(DATA_JSON_PATH)
        # Because of the JSON structure name_english and name is matched
        match_1 = pd.merge(json_ani_img_data, json_ani_data, on='name_english', how='inner')
        match_2 = pd.merge(json_ani_img_data, json_ani_data, left_on='name_english', right_on='name', how='inner')
        concatted_match = pd.concat([match_1, match_2])
        merged_data = concatted_match.drop_duplicates(subset='img')
        merged_data.to_csv(MERGED_CSV_PATH, index=False)
        print(f"Done! --> {MERGED_CSV_PATH}")
    else:
        print(f"It Is Already Done! --> {MERGED_CSV_PATH}")
        merged_data = pd.read_csv(MERGED_CSV_PATH, low_memory=False) 

    # To make sure
    data = merged_data.copy()
    
    # Original index is preserved in file_id without the shifts
    data['file_id'] = data.index
    
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
            # Min Max Normalization
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0

    # Cols added
    JSON_COLS = cols_to_normalize + \
                       [c for c in data.columns if c.startswith('rating_')] + \
                       [c for c in data.columns if c.startswith('Genre_')]
    
    
    
    # Reset index
    anime_data = data.reset_index(drop=True)

    print("\nDownload Image")
    download_all_images(anime_data, DOWNLOADED_IMG_PATH)
    # print("\nImages Downloaded Already")

        # Data clean up (Image Data Check)
    print("Check Images")
    
    if os.path.exists(DOWNLOADED_IMG_PATH):
        existing_files = set(os.listdir(DOWNLOADED_IMG_PATH))
        # Keep only image consisting image
        anime_data = anime_data[anime_data['file_id'].apply(lambda x: f"{x}.jpg" in existing_files)].copy()
        print(f"Cleaned data count --> {len(anime_data)}")
    print("Check Images Done")
    # Check can not catch up
    time.sleep(3)    

    # Train validation
    train_data = anime_data.sample(frac=0.8, random_state=42)
    val_data = anime_data.drop(train_data.index)
    
    # Model
    cnn_model = build_hybrid_model(json_features=len(JSON_COLS))

    train_generator = AnimeDataGenerator(data=train_data, img_dir=DOWNLOADED_IMG_PATH, tab_cols=JSON_COLS, target_cols=TARGET_COLS, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=True)
    val_generator = AnimeDataGenerator(data=val_data, img_dir=DOWNLOADED_IMG_PATH, tab_cols=JSON_COLS, target_cols=TARGET_COLS, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=False)


    # Callbacks 
    early_stop = EarlyStopping(monitor='val_mae', patience=4, restore_best_weights=True, verbose=1) #Early stopping
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_mae', save_best_only=True, verbose=1) # Model save

    # Start
    try:
        history = cnn_model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=[early_stop, checkpoint]
        )

        print("\nTraining Done!")
        
        # Best epoch value
        best_val_mae = min(history.history['val_mae'])
        
    except Exception as e:
        print(f"\nError: {e}")