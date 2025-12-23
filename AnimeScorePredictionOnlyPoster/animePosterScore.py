import pandas as pd
import os
import numpy as np
import cv2 
import time
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Lambda, Dropout, Input, GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision

MODEL_PATH = 'anime_vision_only_model.h5'
MERGED_CSV_PATH = '../data/ani_data_merged.csv'
DOWNLOADED_IMG_PATH = '../data/images/'
EPOCHS = 100 
BATCH_SIZE = 32 
TARGET_COLS = ['score']
TARGET_SIZE = (224, 224, 3)

mixed_precision.set_global_policy('mixed_float16') #RTX3060 prop
class AnimeImageGenerator(Sequence):
    
    # Constructor
    def __init__(self, data, img_dir, target_cols, batch_size, target_size, shuffle): 

        # To not alter the original data
        self.data = data.copy()
        
        # To not get error on indexes after deleting the animes with no poster
        self.data = self.data.reset_index(drop=True) 

        self.img_dir = img_dir # Poster path
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
            self.labels = self.data[self.target_cols].values  # Target data as numpy array

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
        
        return X_img, y 

def build_vision_model():
    # Model
    input_img = Input(shape=TARGET_SIZE)
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_img)
    base_model.trainable = False 
    
    for layer in base_model.layers[-30:]: # Last 30 layers opened for training
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # 7x7x2048 to 1xx2048 dont need the position info
    
    x = Dense(1024, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)       # Stabilize
    x = Activation('relu')(x)         # Activation func
    x = Dropout(0.2)(x)               # To stop memorizing (overfitting) butkeep its weight
    
    x = Dense(512, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)       # Stabilize
    x = Activation('relu')(x)         # Activation func
    x = Dropout(0.15)(x)               # To stop memorizing (overfitting) butkeep its weight
    
    x = Dense(256, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)       # Stabilize
    x = Activation('relu')(x)         # Activation func
    x = Dropout(0.1)(x)               # To stop memorizing (overfitting) butkeep its weight


    output = Dense(1, activation='linear')(x)

    # Keep score between 0-10
    output = Lambda(lambda z: tf.clip_by_value(z, clip_value_min=0.0, clip_value_max=10.0))(output)

    model = Model(inputs=input_img, outputs=output)
    
    optimizer = Adam(learning_rate=0.0002) # Lowered to not break how ResNet50 is
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


if __name__ == "__main__":
    
    # Data already loaded
    df = pd.read_csv(MERGED_CSV_PATH, low_memory=False) 
    
    # To make sure
    data = df.copy()
    
    # To protect indexes after shuffle
    data['file_id'] = data.index
    
    # Data clean up (Image Data Check)
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
    
    # Reset index
    anime_data = data.reset_index(drop=True)


    print("Images are downloaded already.")
    
    # Train validation
    train_data = anime_data.sample(frac=0.8, random_state=42)
    val_data = anime_data.drop(train_data.index)
    
    # Model
    cnn_model = build_vision_model()

    train_generator = AnimeImageGenerator(data=train_data, img_dir=DOWNLOADED_IMG_PATH, target_cols=TARGET_COLS, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=True)
    val_generator = AnimeImageGenerator(data=val_data, img_dir=DOWNLOADED_IMG_PATH, target_cols=TARGET_COLS, batch_size=BATCH_SIZE, target_size=TARGET_SIZE, shuffle=False)
    
    # Callbacks 
    early_stop = EarlyStopping(monitor='val_mae', patience=4, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_mae', save_best_only=True, verbose=1)

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
