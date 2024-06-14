from pathlib import Path
import sys

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import unittest
from fastcg.candidates import CandidatesModel
import pandas as pd
import faiss

import unittest
from unittest.mock import Mock
import numpy as np
import faiss
from fastcg.candidates import CandidatesModel

class TestCandidatesModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        mock_model = Mock()
        mock_model.get_word_id.side_effect = lambda word: 0 if word in ['1', '2', '3'] else -1
        mock_model.get_output_matrix.return_value = np.random.rand(10, 100)

        mock_model.get_input_vector.return_value = np.random.rand(100).astype(np.float32)

        self.model = CandidatesModel(model=mock_model)


    def test_create_faiss_index_for_multiple_words(self):
        words = ['word1', 'word2', 'word3']
        faiss_index = self.model.create_faiss_index_for_multiple_words(words)
        self.assertIsInstance(faiss_index, faiss.IndexFlatIP)

    def test_get_top_similar_to_input_faiss(self):
        word = 'test'
        top_k = 3
        words = ['test', 'word1', 'word2']
        faiss_index = faiss.IndexFlatIP(100)

        input_vector = self.model.model.get_input_vector(0)
        self.assertIsInstance(input_vector, np.ndarray)
        self.assertEqual(input_vector.shape, (100,))

    def test_complementary(self):
        df_input = pd.DataFrame({'id_master': ['1', '2', '3'], 'other_column': [1, 2, 3]})
        candidates = pd.DataFrame({'id': ['1', '2', '3', '4', '5'], 'other_column': [1, 2, 3, 4, 5]})
        nb_candidates = 2
        df_output = self.model.complementary(df_input, candidates, nb_candidates)
        self.assertIsInstance(df_output, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
