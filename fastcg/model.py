import fasttext
import pandas as pd
from .utils import save_model
import csv

class Prod2Vec:
    def __init__(self):
        """
        Initialise l'instance de la classe avec le DataFrame d'entraînement.
        
        :param train_data: DataFrame contenant les colonnes 'produit_1', 'produit_2', 'nb_asso'
        """

    def prepare_data_asso(self, train_data):
        """
        Prépare les données en répétant les lignes selon 'nb_asso' et sauvegarde dans un fichier texte.
        """
        df_repeated = train_data.loc[train_data.index.repeat(train_data['nb_asso'])].reset_index(drop=True)
        df_repeated = df_repeated.drop('nb_asso', axis=1)
        df_repeated.to_csv('train_data.txt', sep=' ', index=False, header=False)

    def prepare_data_session(self, train_data, key_billing, id_art):
        sessions = train_data.groupby(key_billing)[id_art].apply(list).reset_index(name='Session')
        sessions['Session'] = sessions['Session'].apply(lambda x: ' '.join(map(str, x)))
        
        with open('train_data.txt', 'w') as f:
            for session in sessions['Session']:
                f.write(session + '\n')

        return sessions

    def train_model_asso(self, name):
        """
        Entraîne un modèle fastText sur les données préparées.
        """
        try:
            open('train_data.txt', 'r')
        except FileNotFoundError:
            print("Le fichier 'train_data.txt' n'existe pas. Veuillez d'abord exécuter la méthode prepare_data().")
            return

        model = fasttext.train_unsupervised("train_data.txt", neg=50, minCount=1, thread=20)

        # Sauvegarder le modèle
        save_model(model, str(name) + ".bin")

    def run_asso(self):
        """
        Exécute les étapes de préparation des données et d'entraînement du modèle.
        """
        self.prepare_data()
        self.train_model_asso()

    def train_model_session(self, name):
        """
        Entraîne un modèle fastText sur les données préparées.
        """
        try:
            open('train_data.txt', 'r')
        except FileNotFoundError:
            print("Le fichier 'train_data.txt' n'existe pas. Veuillez d'abord exécuter la méthode prepare_data().")
            return

        model = fasttext.train_unsupervised("train_data.txt", neg=50, minCount=1, thread=20)

        # Sauvegarder le modèle
        save_model(model, str(name) + ".bin")
