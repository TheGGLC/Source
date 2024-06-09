import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns
import cupy as cp
import torch
from torch.utils.data import DataLoader

def plot_cupy(class_1_downsampled, class_0_downsampled):
    plt.figure(figsize=(8, 6))
    threshold = 0.05
    class_1_downsampled = cp.asarray(class_1_downsampled)
    class_0_downsampled = cp.asarray(class_0_downsampled)
    # Calculate distances using GPU
    distances = cp.linalg.norm(class_1_downsampled[:, cp.newaxis] - class_0_downsampled, axis=2)

    # Determine which points are within the threshold distance
    overlapping_indices_1, overlapping_indices_0 = cp.where(distances < threshold)

    # Convert results back to NumPy
    overlapping_indices_1 = cp.asnumpy(overlapping_indices_1)
    overlapping_indices_0 = cp.asnumpy(overlapping_indices_0)

    # Mask for non-overlapping points
    non_overlapping_mask_1 = np.ones(len(class_1_downsampled), dtype=bool)
    non_overlapping_mask_1[overlapping_indices_1] = False
    non_overlapping_mask_0 = np.ones(len(class_0_downsampled), dtype=bool)
    non_overlapping_mask_0[overlapping_indices_0] = False

    # Plot non-overlapping points
    plt.scatter(class_1_downsampled[non_overlapping_mask_1][:, 0].get(), class_1_downsampled[non_overlapping_mask_1][:, 1].get(), color='r', label='unplayable', alpha=0.5)
    plt.scatter(class_0_downsampled[non_overlapping_mask_0][:, 0].get(), class_0_downsampled[non_overlapping_mask_0][:, 1].get(), color='b', label='playable', alpha=0.3)

    # Plot overlapping points
    overlapping_points_1 = class_1_downsampled[overlapping_indices_1]
    overlapping_points_0 = class_0_downsampled[overlapping_indices_0]
    plt.scatter(overlapping_points_1[:, 0].get(), overlapping_points_1[:, 1].get(), color='purple', label='overlapping', alpha=0.5)
    plt.scatter(overlapping_points_0[:, 0].get(), overlapping_points_0[:, 1].get(), color='purple', alpha=0.5)

    # Hide x and y axis values
    plt.xticks([])  # Hide x-axis tick labels
    plt.yticks([])  # Hide y-axis tick labels
    plt.tight_layout()
    plt.savefig(f'./embeddings/{game}/{game}_distribution-{method}_new.png', dpi=300)
    plt.close()

def plot_dataset_cupy(embeddings_2d, dataset):
    # Convert embeddings to CuPy array
    embeddings_2d = cp.asarray(embeddings_2d)

    # Prepare for plotting
    plt.figure(figsize=(8, 6))

    for i in range(len(dataset)):
        image, label = dataset[i]
        x = cp.asnumpy(embeddings_2d[i, 0])  # Convert back to NumPy for plotting
        y = cp.asnumpy(embeddings_2d[i, 1])  # Convert back to NumPy for plotting
        plt.scatter(x, y, c=plt.cm.tab10(label / 10.0), label=label, s=5, alpha=0.5)
        # Hide x and y axis values
    plt.xticks([])  # Hide x-axis tick labels
    plt.yticks([])  # Hide y-axis tick labels
    plt.tight_layout()
    plt.savefig(f'./embeddings/{game}/{game}_distribution-{method}_new.png', dpi=300)
    plt.close()

def plot_cupy_batched(class_1_downsampled, class_0_downsampled, batch_size=1000):
    plt.figure(figsize=(8, 6))
    threshold = 0.05
    class_1_downsampled = cp.asarray(class_1_downsampled)
    class_0_downsampled = cp.asarray(class_0_downsampled)
    
    for i in range(0, len(class_1_downsampled), batch_size):
        batch_class_1 = class_1_downsampled[i:i+batch_size]
        distances = cp.linalg.norm(batch_class_1[:, cp.newaxis] - class_0_downsampled, axis=2)
        overlapping_indices_1, _ = cp.where(distances < threshold)
        non_overlapping_mask_1 = np.ones(len(batch_class_1), dtype=bool)
        non_overlapping_mask_1[overlapping_indices_1.get()] = False
        plt.scatter(batch_class_1[non_overlapping_mask_1][:, 0].get(), batch_class_1[non_overlapping_mask_1][:, 1].get(), color='r', label='unplayable', alpha=0.5)
        
        overlapping_points_1 = batch_class_1[overlapping_indices_1]
        plt.scatter(overlapping_points_1[:, 0].get(), overlapping_points_1[:, 1].get(), color='purple', label='overlapping', alpha=0.5)

    for i in range(0, len(class_0_downsampled), batch_size):
        batch_class_0 = class_0_downsampled[i:i+batch_size]
        distances = cp.linalg.norm(class_1_downsampled[:, cp.newaxis] - batch_class_0, axis=2)
        _, overlapping_indices_0 = cp.where(distances < threshold)
        non_overlapping_mask_0 = np.ones(len(batch_class_0), dtype=bool)
        non_overlapping_mask_0[overlapping_indices_0.get()] = False
        plt.scatter(batch_class_0[non_overlapping_mask_0][:, 0].get(), batch_class_0[non_overlapping_mask_0][:, 1].get(), color='b', label='playable', alpha=0.3)

        overlapping_points_0 = batch_class_0[overlapping_indices_0]
        plt.scatter(overlapping_points_0[:, 0].get(), overlapping_points_0[:, 1].get(), color='purple', alpha=0.5)

    plt.xticks([])  # Hide x-axis tick labels
    plt.yticks([])  # Hide y-axis tick labels
    plt.tight_layout()
    plt.savefig(f'./embeddings/{game}/{game}_batched_distribution-{method}_new.png', dpi=300)
    plt.close()

def plot_dataset_cupy_batched(embeddings_2d, dataset, batch_size=1000):
    # Convert embeddings to CuPy array
    embeddings_2d = cp.asarray(embeddings_2d)

    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare for plotting
    plt.figure(figsize=(8, 6))

    for batch_data in data_loader:
        images, labels = batch_data
        batch_embeddings = embeddings_2d[labels]

        for i in range(len(images)):
            x = cp.asnumpy(batch_embeddings[i, 0])  # Convert back to NumPy for plotting
            y = cp.asnumpy(batch_embeddings[i, 1])  # Convert back to NumPy for plotting
            plt.scatter(x, y, c=plt.cm.tab10(labels[i] / 10.0), label=labels[i], s=5, alpha=0.5)
    
    # Hide x and y axis values
    plt.xticks([])  # Hide x-axis tick labels
    plt.yticks([])  # Hide y-axis tick labels
    plt.tight_layout()
    plt.savefig(f'./embeddings/{game}/{game}_distribution-{method}_new.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--game', type=str)
    args = parser.parse_args()
    method = "umap"

    game = args.game
    if game in ["cifar10", "mnist", "fashion_mnist"]:
        import torchvision.transforms as transforms
        embedding = np.load(f'./embeddings/{game}/p_{method}.npy')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if game == "fashion_mnist":
            from torchvision.datasets import FashionMNIST
            dataset = FashionMNIST(root='./dataset', train=True, download=False, transform=transform)
        elif game == "cifar10":
            from torchvision.datasets import CIFAR10
            dataset = CIFAR10(root='./dataset', train=True, download=True, transform=transform)
        elif game == "mnist":
            from torchvision.datasets import MNIST
            dataset = MNIST(root='./dataset', train=True, download=True, transform=transform)

        plot_dataset_cupy_batched(embedding, dataset)
    elif game == "cave":
        embedding_set_1_1 = np.load(f'./embeddings/{game}/p_{method}.npy')
        embedding_set_1_2 = np.load(f'./embeddings/{game}_doors/p_{method}.npy')
        embedding_set_1_3 = np.load(f'./embeddings/{game}_portal/p_{method}.npy')

        embedding_set_2_1 = np.load(f'./embeddings/{game}/up_{method}.npy')
        embedding_set_2_2 = np.load(f'./embeddings/{game}_doors/up_{method}.npy')
        embedding_set_2_3 = np.load(f'./embeddings/{game}_portal/up_{method}.npy')

        embedding_set_2 = np.concatenate((embedding_set_1_1, embedding_set_1_2, embedding_set_1_3), axis=0)
        embedding_set_1 = np.concatenate((embedding_set_2_1, embedding_set_2_2, embedding_set_2_3), axis=0)
        plot_cupy_batched(embedding_set_1, embedding_set_2)
    else:
        embedding_set_2 = np.load(f'./embeddings/{game}/p_{method}.npy')
        embedding_set_1 = np.load(f'./embeddings/{game}/up_{method}.npy')

        plot_cupy(embedding_set_1, embedding_set_2)
    # plot_different(embedding_set_1, embedding_set_2)
