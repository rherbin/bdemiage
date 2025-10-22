from deepface import DeepFace
import os
import json
from user_sorting import display_face_with_list
import numpy as np
from multiprocessing import Process, Queue, set_start_method

"""
Person :
    Name, embedding, number of samples, images
Image : 
    Person, loc
"""

try:
    set_start_method("spawn") 
except RuntimeError:
    pass

# ArcFace = DeepFace.build_model("ArcFace")

try:
    with open('data_persons.json', 'r') as fp:
        persons_data = json.load(fp)
        for p, data in persons_data.items():
            data["emb"] = np.array(data["emb"])
except:
    persons_data = {}

try:
    with open('data_images.json', 'r') as fp:
        images_data = json.load(fp)
except:
    images_data = {}

def get_closest_neighbor(target_emb, emb_dict):
    """
    in : a target embedding and a dict of embeddings
    out : closest neighbor according to cosine similarity
    """
    if not emb_dict:
        return None
    
    target_emb = np.array(target_emb)
    target_norm = target_emb / np.linalg.norm(target_emb)

    scores = {}

    for label, data in emb_dict.items():
        emb = data["emb"]
        emb_norm = emb / np.linalg.norm(emb)
        score = float(np.dot(target_norm, emb_norm))  # cosine similarity
        scores[label] = score

    best_label = max(scores, key=scores.get)

    return best_label


def get_embeddings(image):
    emb = DeepFace.represent(img_path=image, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)
    return emb

def get_embeddings_sub(q, image):
    """
    puts an embedding in a queue, intended to be used in a subprocess
    """
    emb = DeepFace.represent(img_path=image, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)
    q.put(emb)

def process_image(embs, path):
    """
    embs : list of embedding dict
    path : path of the image

    processes the user validation for face labels
    """
    img_name = path.split("/")[-1]
    if img_name in images_data:
        return
    
    for e in embs:
        suggestion = get_closest_neighbor(e["embedding"], emb_dict=persons_data)
        label = display_face_with_list(path, list(persons_data.keys()), suggestion, e["facial_area"])
        if not label:
            continue

        if label not in persons_data:
            persons_data[label] = {
                "emb" : np.array(e["embedding"]),
                "sample_size" : 1,
                "images" : [{
                    "name" : img_name,
                    "location" : e["facial_area"],
                }]
            }
        else:
            persons_data[label]["emb"] = (persons_data[label]["emb"]*persons_data[label]["sample_size"] + e["embedding"])/persons_data[label]["sample_size"]
            persons_data[label]["sample_size"] += 1
            persons_data[label]["images"].append({
                    "name" : img_name,
                    "location" : e["facial_area"],
                })
        
        images_data[img_name] = [{
            "name" : label,
            "location" : e["facial_area"],
        }]

def process_batch(images):
    q = Queue()

    start = 0
    while start < len(images) and images[start].split("/")[-1] in images_data:
        start += 1
    embs = get_embeddings(images[0])

    for i, img in enumerate(images[:-1]):

        img_name = img.split("/")[-1]
        if img_name in images_data:
            continue

        p = Process(target=get_embeddings_sub, args=(q, images[i+1]))
        p.start()

        process_image(embs, img)
        
        p.join()
        embs = q.get()
    if images[-1].split("/")[-1] not in images_data:
        process_image(embs, images[-1])
    # print(persons_data)

def process_folder(path):
    images = []
    for x in os.listdir(path):
        if x.lower().endswith(("png", "jpg")):
            images.append(os.path.join(path, x))
    process_batch(images)

if __name__ == "__main__":
    # process_batch(["public/albums/mic_2025_amiens/100_2038.JPG", "public/albums/mic_2025_amiens/100_2037.JPG", "public/albums/mic_2025_amiens/100_2048.JPG", "public/albums/mic_2025_amiens/100_2066.JPG"])
    process_folder("public/albums/mic_2025_amiens")
    for p, data in persons_data.items():
        data["emb"] = data["emb"].tolist()
    with open('data_persons.json', 'w') as fp:
        json.dump(persons_data, fp)
    with open('data_images.json', 'w') as fp:
        json.dump(images_data, fp)