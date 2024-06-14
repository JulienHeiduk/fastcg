import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import normaliser_vecteurs, creer_index_faiss, extraire_vecteurs_mots
import fasttext as ft

class CandidatesModel:
    def __init__(self, path):
        self.model = ft.load_model(path)

    def create_faiss_index_for_multiple_words(self, words):
        word_vectors = extraire_vecteurs_mots(self.model, words)
        word_vectors = normaliser_vecteurs(word_vectors)
        dimension = word_vectors.shape[1]
        faiss_index = creer_index_faiss(dimension, word_vectors)
        return faiss_index
        
    def get_top_similar_to_input_faiss(self, word1, faiss_index, top_k, words):
        word1_index = self.model.get_word_id(word1)
        if word1_index == -1:
            raise ValueError(f"{word1} is not in the model's dictionary.")
        
        in_vector = np.array(self.model.get_input_vector(word1_index), dtype=np.float32)
        in_vector = normaliser_vecteurs(in_vector.reshape(1, -1))
        
        distances, indices = faiss_index.search(in_vector, top_k)
        top_scores_and_words = [(words[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return top_scores_and_words
    
    def complementary(self, df_input: pd.DataFrame, candidates: pd.DataFrame, nb_candidates: int):
        df_output = pd.DataFrame()
        candidates = candidates.drop_duplicates(subset=['id'])
        candidates['id'] = candidates['id'].astype(str)
        faiss_index = self.create_faiss_index_for_multiple_words(candidates['id'].tolist())
        list_num_art_to_reocs = df_input['id_master'].drop_duplicates().tolist()
                               
        for i in tqdm(list_num_art_to_reocs):
            try:
                word1 = str(i)
                top_results = self.get_top_similar_to_input_faiss(word1, faiss_index, nb_candidates, candidates['id'].tolist())
                ids_score = pd.DataFrame({
                    'id_master': i,
                    'id_candidates': [result[0] for result in top_results],
                    'score': [result[1] for result in top_results],
                    'rank_candidate': list(range(len(top_results)))
                })
                df_output = pd.concat([df_output, ids_score], ignore_index=True)
            except Exception as error:
                print(f"Error processing article {i}: {error}")
                continue

        return df_output
