import fasttext
import numpy as np
import faiss


def save_model(model, chemin: str):
    """Sauvegarde le modèle entraîné."""
    model.save_model(chemin)

def load_model(chemin: str):
    """Charge un modèle entraîné."""
    return fasttext.load_model(chemin)

def normaliser_vecteurs(vecteurs):
    if vecteurs.ndim == 1:
        vecteurs = vecteurs.reshape(1, -1)
    normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
    normes = np.where(normes == 0, 1, normes)
    vecteurs_normalises = vecteurs / normes
    return vecteurs_normalises



def creer_index_faiss(dimension, vecteurs):
    index_faiss = faiss.IndexFlatIP(int(dimension))
    index_faiss.add(vecteurs)
    return index_faiss


def preparer_dataframe_candidates(candidates):
    """Prépare le DataFrame des candidats."""
    candidates['id'] = candidates['id'].drop_duplicates().astype(str)
    return candidates

def extraire_vecteurs_mots(model, mots):
    """Extrait les vecteurs pour une liste de mots à partir du modèle."""
    vecteurs = [model.get_output_matrix()[model.get_word_id(mot)] for mot in mots if model.get_word_id(mot) != -1]
    return np.array(vecteurs, dtype=np.float32)

def create_faiss_index_for_multiple_words(model, words):
    vecteurs = extraire_vecteurs_mots(model, words)
    vecteurs_normalises = normaliser_vecteurs(vecteurs)
    dimension = vecteurs_normalises.shape[1]
    print(dimension)
    faiss_index = creer_index_faiss(dimension, vecteurs_normalises)
    return faiss_index

