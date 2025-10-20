from deepface import DeepFace
import os
import pandas as pd

query_path = "public/albums/mic_2025_amiens/100_2037.JPG"
db_path = "public/albums/mic_2025_amiens/"

records = []

for img_name in os.listdir(db_path):
    img_path = os.path.join(db_path, img_name)
    try:
        result = DeepFace.verify(query_path, img_path, enforce_detection=False)
        if result["verified"]:  # Si DeepFace pense que c'est la même personne
            records.append((img_name, result["distance"]))
            print(img_path)
    except Exception:
        print("Exception")

# Tri par distance (plus petite = plus proche)
df = pd.DataFrame(records, columns=['img', 'distance'])
df = df.sort_values('distance').reset_index(drop=True)

# print(df.head(10))
DeepFace.represent()