import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
     
def radius(n_samples, radius_arr, data, labels):
    all_data = []
    for radius in radius_arr:  # Assuming you have an array of radii to iterate over
        # Sample n data points
        sample_indices = np.random.choice(len(data), n_samples, replace=False)
        sample_data = data[sample_indices]
        sample_labels = labels[sample_indices]

        # Find neighbors within the distance radius
        nbrs = RadiusNeighborsClassifier(radius=radius, algorithm='auto').fit(data, labels)
        distances, indices = nbrs.radius_neighbors(sample_data)

        # Calculate the percentage of neighbors with the opposite label
        percentages = []
        for i in range(n_samples):
            neighbor_indices = indices[i]
            # Exclude the sample itself
            neighbor_indices = neighbor_indices[neighbor_indices != sample_indices[i]]

            if len(neighbor_indices) > 0:
                neighbor_labels = labels[neighbor_indices]
                same_label_count = np.sum(neighbor_labels != sample_labels[i])
                percentage = (same_label_count / len(neighbor_indices)) * 100
            else:
                # If there are no neighbors within the radius, handle this case (e.g., assign 0 percentage)
                percentage = 0
            percentages.append(percentage)

        all_data.append(np.mean(percentages))
    return all_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    plotting_data = []
    plotting_label = []
    games = ['mnist', 'cifar10', 'platform', 'vertical', 'cave' , 'crates', 'slide']
    n_samples = 1000  # Number of samples to take
    k_neighbors_arr = np.arange(0.0001, 0.0011, 0.0001)
    method = 'umap'

    for game in games:
        if game == "cave":
            embedding_set_1_1 = np.load(f'./embeddings/{game}/p_{method}.npy')
            embedding_set_1_2 = np.load(f'./embeddings/{game}_doors/p_{method}.npy')
            embedding_set_1_3 = np.load(f'./embeddings/{game}_portal/p_{method}.npy')

            embedding_set_2_1 = np.load(f'./embeddings/{game}/up_{method}.npy')
            embedding_set_2_2 = np.load(f'./embeddings/{game}_doors/up_{method}.npy')
            embedding_set_2_3 = np.load(f'./embeddings/{game}_portal/up_{method}.npy')

            embedding_set_1 = np.concatenate((embedding_set_1_1, embedding_set_1_2, embedding_set_1_3), axis=0)
            embedding_set_2 = np.concatenate((embedding_set_2_1, embedding_set_2_2, embedding_set_2_3), axis=0)

            data = np.vstack((embedding_set_1, embedding_set_2))
            labels = np.hstack((np.zeros(len(embedding_set_1)), np.ones(len(embedding_set_2))))

            data = np.array(data)
            labels = np.array(labels)
            data, labels = shuffle(data, labels, random_state=42)

        elif game == "mnist":
            from torchvision.datasets import MNIST
            import torchvision.transforms as transforms
            transform = transforms.Compose([
            transforms.ToTensor()
            ])
            dataset = MNIST(root='./dataset', train=True, download=False, transform=transform)
            embeddings_2d = np.load(f'./embeddings/{game}/p_{method}.npy')
            data = []
            labels = []
            for i in range(len(dataset)):
                    image, label = dataset[i]
                    data.append(embeddings_2d[i])
                    labels.append(label)

            data = np.array(data)
            labels = np.array(labels)
            data, labels = shuffle(data, labels, random_state=42)

        elif game == "fashion_mnist":
            from torchvision.datasets import FashionMNIST
            import torchvision.transforms as transforms
            transform = transforms.Compose([
            transforms.ToTensor()
            ])
            dataset = FashionMNIST(root='./dataset', train=True, download=False, transform=transform)
            embeddings_2d = np.load(f'./embeddings/{game}/p_{method}.npy')
            data = []
            labels = []
            for i in range(len(dataset)):
                    image, label = dataset[i]
                    data.append(embeddings_2d[i])
                    labels.append(label)

            data = np.array(data)
            labels = np.array(labels)
            data, labels = shuffle(data, labels, random_state=42)

        elif game == "cifar10":
            from torchvision.datasets import CIFAR10
            import torchvision.transforms as transforms
            transform = transforms.Compose([
            transforms.ToTensor()
            ])
            dataset = CIFAR10(root='./dataset', train=True, download=False, transform=transform)
            embeddings_2d = np.load(f'./embeddings/{game}/p_{method}.npy')

            data = []
            labels = []
            for i in range(len(dataset)):
                    image, label = dataset[i]
                    data.append(embeddings_2d[i])
                    labels.append(label)

            data = np.array(data)
            labels = np.array(labels)
            data, labels = shuffle(data, labels, random_state=42)

        else:
            embedding_set_1 = np.load(f'./embeddings/{game}/p_umap.npy')
            embedding_set_2 = np.load(f'./embeddings/{game}/up_umap.npy')

            data = np.vstack((embedding_set_1, embedding_set_2))
            labels = np.hstack((np.zeros(len(embedding_set_1)), np.ones(len(embedding_set_2))))

            data = np.array(data)
            labels = np.array(labels)
            data, labels = shuffle(data, labels, random_state=42)

        # all_data = with_trails(n_samples, k_neighbors_arr, trials, data, labels)
        min_x = np.min(data[:, 0])
        max_x = np.max(data[:, 0])

        min_y = np.min(data[:, 1])
        max_y = np.max(data[:, 1])

        normalized_dist_x = (data[:, 0] - min_x) / (max_x - min_x)
        normalized_dist_y = (data[:, 1] - min_y) / (max_y - min_y)
        normalized_dist = np.column_stack((normalized_dist_x, normalized_dist_y))
        print(game)
        print(np.min(normalized_dist), np.max(normalized_dist))
        # all_data = without_trails(n_samples, k_neighbors_arr, normalized_dist, labels)
        all_data = radius(n_samples, k_neighbors_arr, normalized_dist, labels)
        # means = [np.mean(data) for data in all_data]
        plotting_data.append(all_data)
        plotting_label.append(game)



    print("TABLE")
    print(" & r=".join(map(lambda x: f"{x:.4f}", k_neighbors_arr)))
    
    plt.figure(figsize=(22, 8))
    i = 0
    for data, label in zip(plotting_data, plotting_label):
        print(f"{label} & ")
        print(" & ".join(map(lambda x: f"{x:.1f}", data)))
        i += 1