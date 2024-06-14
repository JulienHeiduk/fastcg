from setuptools import setup, find_packages

setup(
    name="fastcg",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'fasttext',
        'faiss',
        'tqdm'
    ],
    author="Julien Heiduk",
    author_email="julien.heiduk@gmail.com",
    description="Candidates generator based on Prod2Vec",
    keywords="Recommender system",
)