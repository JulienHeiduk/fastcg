from pathlib import Path
import sys

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import pandas as pd
from fastcg import model, candidates
import fasttext as ft
from itertools import combinations

data_train = pd.read_csv('./examples/data.csv', encoding='ISO-8859-1')

def create_pairs(group):
    pairs = list(combinations(group['StockCode'], 2))
    return pd.DataFrame(pairs, columns=['Produit_1', 'Produit_2'])

pairs_df = data_train.groupby('InvoiceNo').apply(create_pairs).reset_index(drop=True)
nb_asso_df = pairs_df.groupby(['Produit_1', 'Produit_2']).size().reset_index(name='nb_asso')

# Train model
#model = model.Prod2Vec()
#model.prepare_data_asso(train_data = nb_asso_df)
#model.train_model_asso(name = "asso_based")

# Predictions - Associations Based
data_master = {
    'id_master': ['10002'],
}

df_master = pd.DataFrame(data_master)

ids = pd.concat([nb_asso_df['Produit_1'], nb_asso_df['Produit_2']]).drop_duplicates().reset_index(drop=True)
candidates_pool = pd.DataFrame(ids, columns=['id'])

#candidates_gen = candidates.CandidatesModel(path = 'asso_based.bin')
#df_candidates = candidates_gen.complementary(df_input = df_master, candidates = candidates_pool, nb_candidates = 5)
#df_candidates.to_csv("./examples/candidates_asso.csv",index =  False)

# Session based trainer   
model = model.Prod2Vec()
model.prepare_data_session(train_data = data_train, key_billing = 'InvoiceNo', id_art = 'StockCode')
model.train_model_session(name = "session_based")

model_session = ft.load_model("session_based.bin")

def obtenir_vecteur_produit(model, produit):
    return model.get_word_vector(produit)

def produits_similaires(model, produit):
    return model.get_nearest_neighbors(produit)

similaires = produits_similaires(model_session, '22623')
print("Produits similaires à 22623:", similaires)
