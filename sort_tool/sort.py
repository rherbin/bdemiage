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

ArcFace = DeepFace.build_model("ArcFace")

persons_data = {}

def get_closest_neighbor(target_emb, emb_dict):
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
    emb = DeepFace.represent(img_path=image, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)
    q.put(emb)

def process_image(embs, path):
    """
    embs : list of embedding dict
    path : path of the image

    processes the user validation for face labels
    """
    for e in embs:
        suggestion = get_closest_neighbor(e["embedding"], emb_dict=persons_data)
        label = display_face_with_list(path, list(persons_data.keys()), suggestion, e["facial_area"])
        if not label:
            continue
        if label not in persons_data:
            persons_data[label] = {
                "emb" : np.array(e["embedding"]),
                "sample_size" : 1,
            }
        else:
            persons_data[label]["emb"] = (persons_data[label]["emb"]*persons_data[label]["sample_size"] + e["embedding"])/persons_data[label]["sample_size"]
            persons_data[label]["sample_size"] += 1

def process_batch(images):
    q = Queue()

    embs = get_embeddings(images[0])

    for i, img in enumerate(images[:-1]):

        p = Process(target=get_embeddings_sub, args=(q, images[i+1]))
        p.start()

        process_image(embs, img)
        
        p.join()
        embs = q.get()
    process_image(embs, images[-1])
    print(persons_data)

def process_folder(path):
    images = []
    for x in os.listdir(path):
        if x.lower().endswith(("png", "jpg")):
            images.append(os.path.join(path, x))
    process_batch(images)

if __name__ == "__main__":
    # process_batch(["public/albums/mic_2025_amiens/100_2038.JPG", "public/albums/mic_2025_amiens/100_2037.JPG", "public/albums/mic_2025_amiens/100_2048.JPG", "public/albums/mic_2025_amiens/100_2066.JPG"])
    process_folder("public/albums/mic_2025_amiens/test")
