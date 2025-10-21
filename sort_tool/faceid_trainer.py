import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.linear_model import SGDClassifier
import joblib

class FaceLearner:
    def __init__(self, model_name="ArcFace", enforce_detection=False):
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(loss='log', max_iter=1000)
        self.fitted = False
        self.labels_seen = set()

    def embed(self, img_path):
        try:
            emb = DeepFace.represent(img_path=img_path, model_name=self.model_name,
                                     enforce_detection=self.enforce_detection)
            return np.array(emb[0]["embedding"], dtype=np.float32)
        except Exception:
            return None

    def predict_name(self, emb):
        if not self.fitted:
            return "unknown", 0.0
        emb = normalize(emb.reshape(1, -1))
        X = self.scaler.transform(emb)
        probs = self.clf.predict_proba(X)[0]
        idx = np.argmax(probs)
        name = self.le.inverse_transform([idx])[0]
        return name, float(probs[idx])

    def update_model(self, emb, name):
        emb = normalize(emb.reshape(1, -1))
        if not self.fitted:
            # premier apprentissage
            self.scaler.fit(emb)
            X = self.scaler.transform(emb)
            self.le.fit([name])
            y = self.le.transform([name])
            self.clf.partial_fit(X, y, classes=np.arange(1))
            self.fitted = True
            self.labels_seen.add(name)
        else:
            if name not in self.labels_seen:
                all_labels = list(self.labels_seen) + [name]
                self.le.fit(all_labels)
                self.labels_seen.add(name)
            X = self.scaler.transform(emb)
            y = self.le.transform([name])
            self.clf.partial_fit(X, y)

def interactive_learning_loop(img_folder):
    learner = FaceLearner()
    image_paths = [os.path.join(img_folder, f)
                   for f in os.listdir(img_folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for path in image_paths:
        print(f"\n➡️ Image: {path}")
        try:
            detections = DeepFace.detectFace(img_path=path, detector_backend="retinaface",
                                             enforce_detection=False, align=False)
        except Exception:
            detections = []

        # si detectFace retourne une seule image (non liste)
        if isinstance(detections, np.ndarray):
            detections = [detections]

        for i, face_img in enumerate(detections):
            # sauvegarde temporaire du visage pour embedding
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, (face_img * 255).astype("uint8"))

            emb = learner.embed(temp_path)
            if emb is None:
                continue

            name, prob = learner.predict_name(emb)
            print(f"Proposition: {name} ({prob:.2f})")

            # afficher visage
            cv2.imshow("Visage", (face_img * 255).astype("uint8"))
            key = input("Nom correct ? (Entrée = valider, ou tapez le nom) : ")

            if key.strip() == "":
                print(f"✅ Confirmé : {name}")
                learner.update_model(emb, name)
            else:
                print(f"🔁 Correction : {key}")
                learner.update_model(emb, key.strip())

            cv2.destroyAllWindows()

    joblib.dump(learner, "face_learner.joblib")
    print("✅ Enregistrement terminé.")

if __name__ == "__main__":
    interactive_learning_loop("public/albums/mic_2025_amiens/")