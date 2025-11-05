#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamic Hypergraph Model for CIFAR-100 Classification
Batch-compatible version matching Dynamic_Hypergraph.ipynb
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch_geometric.nn import HypergraphConv, AttentionalAggregation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch jobs
import matplotlib.pyplot as plt
import os

print("Starting Dynamic Hypergraph Model for CIFAR-100 Classification")

def image_to_dynamic_hypergraph(images, k_spatial=4, k_feature=4):
    batch_node_feats = []
    batch_edge_index = []
    batch_map = []
    node_offset = 0

    for b, img in enumerate(images):
        # img: [C,H,W] -> patches
        C, H, W = img.shape
        patch_size = 8  # 8x8 patches for CIFAR-10
        patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(C, -1, patch_size, patch_size)
        patches = patches.view(patches.size(1), -1)  # flatten each patch to vector

        num_nodes = patches.size(0)
        node_feats = patches

        # Spatial edges (static)
        spatial_edges = []
        coords = np.array([[i // (W // patch_size), i % (W // patch_size)] for i in range(num_nodes)])
        for i in range(num_nodes):
            dists = np.sum((coords - coords[i])**2, axis=1)
            nn_idx = np.argsort(dists)[1:k_spatial+1]
            for j in nn_idx:
                spatial_edges.append([i, j])

        # Feature hyperedges (dynamic, computed on GPU)
        feats = node_feats
        feats = feats.to(images.device)
        dists_feat = torch.cdist(feats.float(), feats.float(), p=2)
        feature_edges = []
        for i in range(num_nodes):
            nn_idx = torch.topk(dists_feat[i], k=k_feature+1, largest=False).indices[1:]
            for j in nn_idx:
                feature_edges.append([i, j])

        all_edges = torch.tensor(spatial_edges + feature_edges, dtype=torch.long, device=images.device).T
        batch_node_feats.append(node_feats)
        batch_edge_index.append(all_edges + node_offset)
        batch_map.append(torch.full((num_nodes,), b, dtype=torch.long, device=images.device))

        node_offset += num_nodes

    x = torch.cat(batch_node_feats, dim=0).float()
    edge_index = torch.cat(batch_edge_index, dim=1)
    batch_map = torch.cat(batch_map)
    return x, edge_index, batch_map

class HyperVigClassifier(nn.Module):
    def __init__(self, in_channels, hidden, num_classes):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, hidden)
        self.conv2 = HypergraphConv(hidden, hidden)
        self.conv3 = HypergraphConv(hidden, hidden)
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch_map):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        out = self.pool(x, batch_map)  # attention pooling
        out = self.classifier(out)
        return out

def main():
    # Data loading
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    print("Loading CIFAR-100 dataset...")
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    print(f"Train dataset shape: {train_dataset.data.shape}")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = HyperVigClassifier(in_channels=3*8*8, hidden=256, num_classes=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Lists to store training history
    train_accuracies = []
    train_losses = []

    # Training loop
    num_epochs = 300
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            node_feats, edge_index, batch_map = image_to_dynamic_hypergraph(images)
            node_feats, edge_index, batch_map = node_feats.to(device), edge_index.to(device), batch_map.to(device)
            
            optimizer.zero_grad()
            outputs = model(node_feats, edge_index, batch_map)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.5f}, Accuracy: {epoch_accuracy:.5f}")

    # Print final results
    print(f"\nFinal Training Accuracy: {train_accuracies[-1]:.4f} ({train_accuracies[-1]*100:.2f}%)")
    print(f"Best Training Accuracy: {max(train_accuracies):.4f} ({max(train_accuracies)*100:.2f}%) at Epoch {train_accuracies.index(max(train_accuracies)) + 1}")

    # Plot Accuracy vs Epochs
    print("\nGenerating training plot...")
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, 'b-o', linewidth=2, markersize=6, label='Training Accuracy')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Training Accuracy vs Epochs - Dynamic Hypergraph Model', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.ylim([0, 1])
    plt.xlim([1, len(epochs)])

    # Add value annotations on the plot
    for i, acc in enumerate(train_accuracies):
        if i % 2 == 0 or i == len(train_accuracies) - 1:  # Show every 2nd epoch and last epoch
            plt.annotate(f'{acc:.3f}', 
                        (epochs[i], acc), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        fontsize=9)

    plt.tight_layout()
    
    # Save plot instead of showing (for batch jobs)
    plot_path = 'training_accuracy_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
