import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import umap
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--game', type=str)
    args = parser.parse_args()

    game = args.game
    embedding_set_1 = np.load(f'./embeddings/{game}/p_clip.npy')

    umap_model = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d_1 = umap_model.fit_transform(embedding_set_1)

    np.save(f'./embeddings/{game}/p_umap.npy', np.array(embeddings_2d_1))

    embedding_set_2 = np.load(f'./embeddings/{game}/up_clip.npy')
    umap_model = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d_2 = umap_model.fit_transform(embedding_set_2)

    np.save(f'./embeddings/{game}/up_umap.npy', np.array(embeddings_2d_2))
    