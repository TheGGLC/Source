import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import PIL
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
import glob
import numpy as np



def find_png_files(directory):
    png_files = []
    search_pattern = os.path.join(directory, '*.png')
    png_files.extend(glob.glob(search_pattern))
    return png_files

def get_embedding(image_paths):
    print("embedding...")
    image_embeddings = []
    for image_path in image_paths:
        # Load and preprocess the image
        try:
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                image_embeddings.append(image_features[0].cpu().numpy())

        except PIL.UnidentifiedImageError:
            continue
        
            
    return image_embeddings 
    
def truncate_to_same_length(list1, list2):
    min_length = min(len(list1), len(list2))
    if min_length > 0:
        return list1[:min_length], list2[:min_length]
    else:
        return list1, list2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--game', type=str)
    args = parser.parse_args()
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA device count: ", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    print(f"device = {device}")
    game = args.game
    game_directory_p = f'./game_dataset/db_{game}/playable/images'
    image_pathes_p = find_png_files(game_directory_p)
    print(f"{len(image_pathes_p)} playable levels found.")

    if game == "platform":
        game_directory_up = f'./game_dataset/db_{game}/unplayable/images'
        image_pathes_up_1 = find_png_files(game_directory_up)
        game_directory_up = f'./game_dataset/db_{game}/unsuable_corrected/images'
        image_pathes_up_2 = find_png_files(game_directory_up)
        image_pathes_up = np.concatenate((image_pathes_up_1, image_pathes_up_2), axis=0)
    else:
        game_directory_up = f'./game_dataset/db_{game}/unplayable/images'
        image_pathes_up = find_png_files(game_directory_up)
    print(f"{len(image_pathes_up)} unplayable levels found.")

    # list1, list2 = truncate_to_same_length(image_pathes_p, image_pathes_up)
    # print(f"{len(list1)} playable levels.")
    # print(f"{len(list2)} unplayable levels.")

    if not os.path.exists(f'./embeddings/{game}'):
        os.makedirs(f'./embeddings/{game}')

    embedding_set_1 = get_embedding(image_pathes_p)

    np.save(f'./embeddings/{game}/p_clip.npy', embedding_set_1)
    print(f"saved ./embeddings/{game}/p_clip.npy")

    embedding_set_2 = get_embedding(image_pathes_up)

    np.save(f'./embeddings/{game}/up_clip.npy', embedding_set_2)
    print(f"saved ./embeddings/{game}/up_clip.npy")
