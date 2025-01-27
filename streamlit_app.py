import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import os
import cv2
import pickle
from deepface import DeepFace
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from deepface.modules.verification import find_distance

# Paramètres globaux
MODEL_NAME = "Facenet512"
METRICS = [{"cosine": 0.30}, {"euclidean": 20.0}, {"euclidean_l2": 0.78}]


# Fonction 1 : Chargement des embeddings
@st.cache_resource
def load_embeddings(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Fichier d'embeddings introuvable. Vérifiez le chemin.")
        return None


# Calcul de la distance cosine
def cosine_distance(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    return 1 - (dot_product / (norm1 * norm2))

# Calcul de la distance euclidienne
def euclidean_distance(emb1, emb2):
    return np.linalg.norm(np.array(emb1) - np.array(emb2))

# Calcul de la distance euclidienne L2 normalisée
def euclidean_l2_distance(emb1, emb2):
    emb1 = np.array(emb1) / np.linalg.norm(emb1)
    emb2 = np.array(emb2) / np.linalg.norm(emb2)
    return np.linalg.norm(emb1 - emb2)


# Fonction 2 : Détection faciale en temps réel via webcam
def real_time_face_recognition():
    st.subheader("Reconnaissance faciale en temps réel")
    st.write("Utilisez votre webcam pour détecter et reconnaître des visages.")

    # Charger les embeddings
    embs = load_embeddings("./embeddings/embs_facenet512.pkl")
    if embs is None:
        return

    # Conteneur d'affichage
    FRAME_WINDOW = st.image([])
    start_detection = st.button("Démarrer la détection")
    stop_detection = st.button("Arrêter la détection")

    # Variable pour mémoriser l'état du visage détecté
    last_detected_face = None

    if start_detection:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Erreur : Impossible d'ouvrir la caméra.")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur : Impossible de lire le flux vidéo.")
                break

            # Traitement toutes les 5 frames
            if frame_count % 5 == 0:
                detected_faces = []
                results = DeepFace.extract_faces(
                    frame, detector_backend="yolov8", enforce_detection=False
                )

                for result in results:
                    if result["confidence"] >= 0.5:
                        x, y, w, h = (
                            result["facial_area"]["x"],
                            result["facial_area"]["y"],
                            result["facial_area"]["w"],
                            result["facial_area"]["h"],
                        )
                        cropped_face = frame[y : y + h, x : x + w]
                        cropped_face_resized = cv2.resize(cropped_face, (224, 224))
                        cropped_face_rgb = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2RGB)

                        emb = DeepFace.represent(
                            cropped_face_rgb,
                            model_name=MODEL_NAME,
                            enforce_detection=False,
                            detector_backend="skip",
                        )[0]["embedding"]

                        min_dist = float("inf")
                        match_name = None
                        for name, emb2 in embs.items():
                            dst = find_distance(emb, emb2, list(METRICS[2].keys())[0])
                            if dst < min_dist:
                                min_dist = dst
                                match_name = name

                        color = (0, 255, 0) if min_dist < list(METRICS[2].values())[0] else (0, 0, 255)
                        label = f"{match_name if match_name else 'Inconnu'} ({min_dist:.2f})"

                        # Si le visage détecté est différent de celui précédent, mettre à jour la variable
                        if last_detected_face != (x, y, w, h, match_name):
                            last_detected_face = (x, y, w, h, match_name)

                        # Dessiner le cadre permanent
                        if last_detected_face:
                            x, y, w, h, name = last_detected_face
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Affichage
            FRAME_WINDOW.image(frame, channels="BGR")

            # Arrêt
            if stop_detection:
                break
            frame_count += 1

        cap.release()
        st.success("Caméra arrêtée.")


# Fonction 3 : Traitement des images et extraction d'embeddings
def batch_face_processing():
    st.subheader("Traitement par lot et extraction des embeddings")
    input_dir = st.text_input("Dossier d'images d'entrée", "./data")
    output_dir = st.text_input("Dossier de sortie pour les visages extraits", "./cropped_faces")
    emb_file = "embs_facenet512.pkl"
    norm_dir = "./norm_faces"

    if st.button("Lancer le traitement par lot"):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(norm_dir, exist_ok=True)

        embs = {}
        if os.path.exists(f"./{output_dir}/{emb_file}"):
            with open(f"./{output_dir}/{emb_file}", "rb") as file:
                embs = pickle.load(file)

        for img_file in tqdm(os.listdir(input_dir)):
            img_path = os.path.join(input_dir, img_file)
            img_name = img_file.split(".")[0]

            if img_name not in embs:
                face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if face is None:
                    continue

                # Prétraitement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                face_norm = clahe.apply(face)
                face_norm_rgb = cv2.cvtColor(face_norm, cv2.COLOR_GRAY2RGB)
                plt.imsave(f"{norm_dir}/{img_name}.jpg", face_norm_rgb)

                # Embedding
                emb = DeepFace.represent(
                    face_norm_rgb, model_name=MODEL_NAME, enforce_detection=False, detector_backend="skip"
                )[0]["embedding"]
                embs[img_name] = emb

        with open(f"./{output_dir}/{emb_file}", "wb") as file:
            pickle.dump(embs, file)
            st.success("Traitement terminé. Embeddings mis à jour.")


# Interface principale avec les onglets
def main():
    st.title("Application de Reconnaissance Faciale")
    tab1, tab2 = st.tabs(["Reconnaissance en temps réel", "Traitement par lot"])
    
    with tab1:
        real_time_face_recognition()

    with tab2:
        batch_face_processing()


if __name__ == "__main__":
    main()
