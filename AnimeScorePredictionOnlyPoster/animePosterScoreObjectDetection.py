import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from ultralytics import YOLO
from tqdm import tqdm

IMG_DIR = '../data/images/'
MERGED_CSV_PATH = '../data/ani_data_merged.csv'  
OBJECTS_CSV_PATH = '../data/anime_all_objects_detected.csv'  
MIN = 20

def detect_objects_with_yolo():
    

    df = pd.read_csv(MERGED_CSV_PATH, low_memory=False)
    # Make sure
    data = df.copy()

    # File Id add if not
    if 'file_id' not in data.columns:
        data['file_id'] = data.index

    # Filter the ones with no image
    if os.path.exists(IMG_DIR):
        existing_files = set(os.listdir(IMG_DIR))
        data = data[data['file_id'].apply(lambda x: f"{x}.jpg" in existing_files)].copy()

    # Model
    model = YOLO('yolov8m.pt')
    results_list = []

    for index, row in tqdm(data.iterrows(), total=len(data)):
        img_path = os.path.join(IMG_DIR, f"{row['file_id']}.jpg")
        
        counts = {'file_id': row['file_id']}
        
        try:
            # Predict
            results = model.predict(img_path, verbose=False, conf=0.25)
            
            # Count the objects
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                if class_name not in counts:
                    counts[class_name] = 0
                counts[class_name] += 1
                
        except Exception as e:
            pass
        
        results_list.append(counts)

    # Res
    results_df = pd.DataFrame(results_list).fillna(0)
    
    # Convert all to int (to display)
    if not results_df.empty:
        cols = results_df.columns.drop('file_id')
        results_df[cols] = results_df[cols].astype(int)

    results_df.to_csv(OBJECTS_CSV_PATH, index=False)


def analyze_object_impact():
    
    df_main = pd.read_csv(MERGED_CSV_PATH, low_memory=False)
    df_obj = pd.read_csv(OBJECTS_CSV_PATH)
    
    # Score clean up
    df_main = df_main.dropna(subset=['score'])
    df_main['score'] = pd.to_numeric(df_main['score'], errors='coerce')
    df_main = df_main.dropna(subset=['score'])
    
    # File ID check
    if 'file_id' not in df_main.columns:
        df_main['file_id'] = df_main.index
    
    # Merge
    df = pd.merge(df_main[['file_id', 'score', 'name']], df_obj, on='file_id', how='inner')
    
    # Obj cols
    object_columns = [col for col in df.columns if col not in ['file_id', 'score', 'name']]
    
    results = []
    
    for obj in object_columns:
        # Check Object Existence
        the_animes_with = df[df[obj] > 0]
        the_animes_without = df[df[obj] == 0]
        
        count = len(the_animes_with)
        
        if count < MIN:
            continue
            
        mean_with = the_animes_with['score'].mean()
        mean_without = the_animes_without['score'].mean()
        
        # Impact
        impact = mean_with - mean_without
        
        results.append({
            'Object': obj,
            'Count': count,
            'Mean With': mean_with,
            'Mean Without': mean_without,
            'Impact': impact
        })

    # Res
    res_df = pd.DataFrame(results)
    
    top_positive = res_df.sort_values(by='Impact', ascending=False).head(15)
    top_negative = res_df.sort_values(by='Impact', ascending=True).head(15)
    
    print('Best Ones:')
    print(top_positive[['Object', 'Count', 'Mean With', 'Mean Without']].to_string(index=False))
    
    print('Worst Ones:')
    print(top_negative[['Object', 'Count', 'Mean With', 'Mean Without']].to_string(index=False))

    # Plot
    plt.figure(figsize=(14, 8))
    plot_data = pd.concat([top_positive, top_negative])
    
    colors = ["#0dff00" if x > 0 else "#ff0000" for x in plot_data['Impact']]
    
    sns.barplot(x='Impact', y='Object', data=plot_data, palette=colors)
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.title(f"Impacts of objects: {MIN} count at least")
    plt.xlabel("Difference")
    plt.ylabel("Object")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Do once then comment
    detect_objects_with_yolo()
    
    analyze_object_impact()