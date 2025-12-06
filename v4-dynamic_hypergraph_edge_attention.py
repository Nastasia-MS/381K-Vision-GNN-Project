#!/usr/bin/env python3
"""
Dynamic Hypergraph Edge Attention v4
Converted from Jupyter notebook for training on TACC
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch_geometric.nn import HypergraphConv, AttentionalAggregation
import argparse
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH/TACC
import matplotlib.pyplot as plt

# Try to import Neptune, but continue if not available
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("Warning: Neptune not installed. Continuing without Neptune monitoring.")


# ============================================================================
# Hypergraph Construction Functions
# ============================================================================

def image_to_adaptive_hypergraph(images, k_spatial, k_feature=None):
    """
    Build hypergraph with specified k values (can vary per layer).
    If k_feature is None, only use spatial edges.
    """
    batch_node_feats = []
    batch_hyperedge_index = []
    batch_map = []
    node_offset = 0
    edge_id = 0

    for b, img in enumerate(images):
        C, H, W = img.shape
        patch_size = 8

        # Extract patches
        patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, C, patch_size, patch_size)
        node_feats = patches.view(patches.size(0), -1).to(images.device)
        num_nodes = node_feats.size(0)

        # Spatial coordinates
        num_patches_side = W // patch_size
        coords = torch.tensor([
            [i // num_patches_side, i % num_patches_side]
            for i in range(num_nodes)
        ], device=images.device, dtype=torch.float)

        dists_spatial = torch.cdist(coords, coords, p=2)

        hyperedge_list = []

        # Spatial hyperedges
        for i in range(num_nodes):
            nn_idx = torch.topk(dists_spatial[i], k=k_spatial+1, largest=False).indices
            for node in nn_idx:
                hyperedge_list.append([node.item() + node_offset, edge_id])
            edge_id += 1

        # Feature hyperedges (optional, for later layers)
        if k_feature is not None and k_feature > 0:
            dists_feat = torch.cdist(node_feats, node_feats, p=2)
            for i in range(num_nodes):
                nn_idx = torch.topk(dists_feat[i], k=k_feature+1, largest=False).indices
                for node in nn_idx:
                    hyperedge_list.append([node.item() + node_offset, edge_id])
                edge_id += 1

        batch_node_feats.append(node_feats)
        batch_hyperedge_index.extend(hyperedge_list)
        batch_map.append(torch.full((num_nodes,), b, dtype=torch.long, device=images.device))
        node_offset += num_nodes

    x = torch.cat(batch_node_feats, dim=0).float()
    hyperedge_index = torch.tensor(batch_hyperedge_index, dtype=torch.long, device=images.device).T
    batch_map = torch.cat(batch_map)

    return x, hyperedge_index, batch_map


def image_to_true_hypergraph(images, k_spatial=4, k_feature=4):
    """Build hypergraph with both spatial and feature-based hyperedges"""
    batch_node_feats = []
    batch_hyperedge_index = []
    batch_map = []
    node_offset = 0
    edge_id = 0

    for b, img in enumerate(images):
        C, H, W = img.shape
        patch_size = 8

        # Extract patches
        patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, C, patch_size, patch_size)
        node_feats = patches.view(patches.size(0), -1).to(images.device)
        num_nodes = node_feats.size(0)

        # Spatial coordinates for spatial hyperedges
        coords = torch.tensor([
            [i // (W // patch_size), i % (W // patch_size)]
            for i in range(num_nodes)
        ], device=images.device, dtype=torch.float)

        dists_spatial = torch.cdist(coords, coords, p=2)

        # Feature distances for feature-based hyperedges
        dists_feat = torch.cdist(node_feats, node_feats, p=2)

        hyperedge_list = []

        # Spatial hyperedges: for each node, create a hyperedge with its k nearest spatial neighbors
        for i in range(num_nodes):
            nn_idx = torch.topk(dists_spatial[i], k=k_spatial+1, largest=False).indices
            # Create hyperedge: all these nodes belong to one hyperedge
            for node in nn_idx:
                hyperedge_list.append([node.item() + node_offset, edge_id])
            edge_id += 1

        # Feature hyperedges: for each node, create a hyperedge with its k nearest feature neighbors
        for i in range(num_nodes):
            nn_idx = torch.topk(dists_feat[i], k=k_feature+1, largest=False).indices
            for node in nn_idx:
                hyperedge_list.append([node.item() + node_offset, edge_id])
            edge_id += 1

        batch_node_feats.append(node_feats)
        batch_hyperedge_index.extend(hyperedge_list)
        batch_map.append(torch.full((num_nodes,), b, dtype=torch.long, device=images.device))
        node_offset += num_nodes

    x = torch.cat(batch_node_feats, dim=0).float()
    # Convert to PyG hypergraph format: [2, num_edges] where row 0 is nodes, row 1 is hyperedge IDs
    hyperedge_index = torch.tensor(batch_hyperedge_index, dtype=torch.long, device=images.device).T
    batch_map = torch.cat(batch_map)

    return x, hyperedge_index, batch_map


# ============================================================================
# Attention Modules
# ============================================================================

class MultiHeadHyperedgeAttention(nn.Module):
    """Multi-head attention for hyperedges - different heads learn different relationship types"""
    def __init__(self, in_dim, hidden=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        # Each head has its own MLP
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.head_dim, 1)
            ) for _ in range(num_heads)
        ])

        self._init_weights()

    def _init_weights(self):
        for head in self.attention_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        """
        x: [num_nodes, feature_dim]
        edge_index: [2, num_connections] - format: [node_ids, hyperedge_ids]
        Returns: edge weights [num_unique_hyperedges]
        """
        node_idx, hyperedge_idx = edge_index
        num_hyperedges = hyperedge_idx.max().item() + 1

        # Aggregate node features for each hyperedge (mean pooling)
        hyperedge_feats = torch.zeros(num_hyperedges, x.size(1), device=x.device, dtype=x.dtype)
        hyperedge_feats.index_add_(0, hyperedge_idx, x[node_idx])

        # Count nodes per hyperedge for proper averaging
        counts = torch.zeros(num_hyperedges, device=x.device, dtype=x.dtype)
        counts.index_add_(0, hyperedge_idx, torch.ones_like(node_idx, dtype=x.dtype))
        hyperedge_feats = hyperedge_feats / counts.unsqueeze(1).clamp(min=1)

        # Compute attention with each head
        head_weights = []
        for head in self.attention_heads:
            alpha = head(hyperedge_feats).squeeze(-1)
            alpha = torch.clamp(alpha, min=-5, max=5)
            head_weights.append(torch.sigmoid(alpha))

        # Average across heads
        avg_weight = torch.stack(head_weights).mean(0)

        # Scale to [0.1, 1.0] range
        return avg_weight * 0.9 + 0.1


class HyperedgeAttention(nn.Module):
    """Learnable hyperedge attention that's part of the model's forward pass"""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        """
        Compute attention weights for each hyperedge based on aggregated node features.

        x: [num_nodes, feature_dim]
        edge_index: [2, num_connections] - format: [node_ids, hyperedge_ids]
        Returns: edge weights [num_unique_hyperedges]
        """
        node_idx, hyperedge_idx = edge_index
        num_hyperedges = hyperedge_idx.max().item() + 1

        # Aggregate node features for each hyperedge (mean pooling)
        hyperedge_feats = torch.zeros(num_hyperedges, x.size(1), device=x.device, dtype=x.dtype)

        # Use scatter_mean to aggregate features
        hyperedge_feats.index_add_(0, hyperedge_idx, x[node_idx])

        # Count nodes per hyperedge for proper averaging
        counts = torch.zeros(num_hyperedges, device=x.device, dtype=x.dtype)
        counts.index_add_(0, hyperedge_idx, torch.ones_like(node_idx, dtype=x.dtype))
        hyperedge_feats = hyperedge_feats / counts.unsqueeze(1).clamp(min=1)

        # Compute attention scores
        alpha = self.mlp(hyperedge_feats).squeeze(-1)
        alpha = torch.clamp(alpha, min=-5, max=5)

        # prevent zero weights
        return torch.sigmoid(alpha) * 0.9 + 0.1


# ============================================================================
# Hypergraph Blocks
# ============================================================================

class HypergraphBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3, ffn_expansion=2):
        super().__init__()
        self.conv = HypergraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x, edge_index, edge_weight):
        # Hypergraph conv w/ residual
        x_res = x
        x = self.conv(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = self.dropout(F.relu(x)) + x_res

        # Feedforward network w/ residual
        x_res = x
        x = self.ffn(x)
        x = self.norm2(x)
        x = self.dropout(x) + x_res

        return x


class AdaptiveHypergraphBlock(nn.Module):
    """Hypergraph block that can use different k values"""
    def __init__(self, hidden_dim, dropout=0.3, ffn_expansion=2):
        super().__init__()
        self.conv = HypergraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x, edge_index, edge_weight):
        # Hypergraph convolution with residual
        x_res = x
        x = self.conv(x, edge_index, edge_weight)
        x = self.norm1(x)
        x = self.dropout(F.relu(x)) + x_res

        # Feedforward with residual
        x_res = x
        x = self.ffn(x)
        x = self.norm2(x)
        x = self.dropout(x) + x_res

        return x


class HyperedgeToHyperedgeLayer(nn.Module):
    """Allow hyperedges to exchange information with each other"""
    def __init__(self, hidden_dim):
        super().__init__()
        # MLP for hyperedge message passing
        self.hyperedge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_weights):
        """
        x: node features [num_nodes, hidden]
        edge_index: [2, num_connections]
        edge_weights: [num_hyperedges]

        Returns: updated edge_weights
        """
        node_idx, hyperedge_idx = edge_index
        num_hyperedges = hyperedge_idx.max().item() + 1

        # Aggregate node features per hyperedge
        hyperedge_feats = torch.zeros(num_hyperedges, x.size(1), device=x.device, dtype=x.dtype)
        hyperedge_feats.index_add_(0, hyperedge_idx, x[node_idx])
        counts = torch.zeros(num_hyperedges, device=x.device, dtype=x.dtype)
        counts.index_add_(0, hyperedge_idx, torch.ones_like(node_idx, dtype=x.dtype))
        hyperedge_feats = hyperedge_feats / counts.unsqueeze(1).clamp(min=1)

        # Build dual graph: hyperedges that share nodes are connected
        # Two hyperedges are neighbors if they share at least one node
        dual_edges = self.build_dual_graph(edge_index, num_hyperedges)

        # Message passing between hyperedges
        hyperedge_feats_updated = hyperedge_feats.clone()
        for src, dst in dual_edges.T:
            # Hyperedge src sends message to hyperedge dst
            message = self.hyperedge_mlp(hyperedge_feats[src])
            hyperedge_feats_updated[dst] += 0.1 * message  # Small update

        # Update edge weights based on new hyperedge features
        edge_weights_updated = hyperedge_feats_updated.norm(dim=1)
        edge_weights_updated = edge_weights_updated / edge_weights_updated.max()  # Normalize
        edge_weights_updated = edge_weights_updated * 0.9 + 0.1  # Scale to [0.1, 1.0]

        return edge_weights_updated

    def build_dual_graph(self, edge_index, num_hyperedges):
        """Build graph where hyperedges that share nodes are connected"""
        node_idx, hyperedge_idx = edge_index

        # For each node, find which hyperedges it belongs to
        node_to_hyperedges = {}
        for node, hedge in zip(node_idx.tolist(), hyperedge_idx.tolist()):
            if node not in node_to_hyperedges:
                node_to_hyperedges[node] = []
            node_to_hyperedges[node].append(hedge)

        # Build edges between hyperedges that share nodes
        dual_edges = set()
        for node, hedges in node_to_hyperedges.items():
            # Connect all pairs of hyperedges that share this node
            for i in range(len(hedges)):
                for j in range(i+1, len(hedges)):
                    dual_edges.add((hedges[i], hedges[j]))
                    dual_edges.add((hedges[j], hedges[i]))  # Bidirectional

        if len(dual_edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        return torch.tensor(list(dual_edges), dtype=torch.long, device=edge_index.device).T


class HypergraphBlockWithH2H(nn.Module):
    """Hypergraph block with hyperedge-to-hyperedge communication"""
    def __init__(self, hidden_dim, dropout=0.3, ffn_expansion=2):
        super().__init__()
        self.conv = HypergraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        # Hyperedge-to-hyperedge layer
        self.h2h = HyperedgeToHyperedgeLayer(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x, edge_index, edge_weight):
        # Update edge weights via hyperedge communication
        edge_weight_updated = self.h2h(x, edge_index, edge_weight)

        # Hypergraph convolution with updated weights
        x_res = x
        x = self.conv(x, edge_index, edge_weight_updated)
        x = self.norm1(x)
        x = self.dropout(F.relu(x)) + x_res

        # Feedforward
        x_res = x
        x = self.ffn(x)
        x = self.norm2(x)
        x = self.dropout(x) + x_res

        return x


# ============================================================================
# Model Classes
# ============================================================================

class AdaptiveHyperVigClassifier(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, k_schedule=[12, 10, 8, 6, 4, 4], dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.k_schedule = k_schedule  # k values for each layer

        # Project patch features
        self.input_proj = nn.Linear(in_channels, hidden)

        # Multi-head edge attention (shared across layers or per-layer)
        self.edge_attns = nn.ModuleList([
            MultiHeadHyperedgeAttention(in_dim=hidden, hidden=64, num_heads=8)
            for _ in k_schedule
        ])

        # Hypergraph blocks
        self.blocks = nn.ModuleList([
            AdaptiveHypergraphBlock(hidden, dropout=dropout)
            for _ in k_schedule
        ])

        # Attentional pooling
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images, batch_map_input):
        # Extract initial patch features
        batch_size = images.size(0)
        patches_per_image = 16

        # Initial projection
        x_init, _, batch_map = image_to_adaptive_hypergraph(images, k_spatial=self.k_schedule[0])
        x = self.input_proj(x_init)

        # Process through layers with evolving topology
        for layer_idx, (k_val, block, edge_attn) in enumerate(zip(self.k_schedule, self.blocks, self.edge_attns)):
            # Rebuild hypergraph with current k value
            # Use feature-based edges for later layers (when features are better)
            k_feature = k_val if layer_idx >= 2 else None  # Feature edges only in layers 3+

            # Reshape x back to images to rebuild graph
            x_reshaped = x.view(batch_size, patches_per_image, -1)

            # Build new hypergraph based on current features
            _, edge_index, _ = self.build_hypergraph_from_features(
                x_reshaped, k_spatial=k_val, k_feature=k_feature
            )

            # Compute edge attention
            edge_weight = edge_attn(x, edge_index)

            # Apply block
            x = block(x, edge_index, edge_weight)

        # Pool and classify
        out = self.pool(x, batch_map)
        out = self.classifier(out)
        return out

    def build_hypergraph_from_features(self, x_batched, k_spatial, k_feature=None):
        """Helper to rebuild hypergraph from current node features"""
        batch_size, num_nodes, feat_dim = x_batched.shape
        batch_hyperedge_index = []
        node_offset = 0
        edge_id = 0

        for b in range(batch_size):
            x_nodes = x_batched[b]  # [num_nodes, feat_dim]

            # Spatial distances (based on patch coordinates, fixed)
            num_patches_side = 4  # 32/8 = 4
            coords = torch.tensor([
                [i // num_patches_side, i % num_patches_side]
                for i in range(num_nodes)
            ], device=x_nodes.device, dtype=torch.float)
            dists_spatial = torch.cdist(coords, coords, p=2)

            hyperedge_list = []

            # Spatial hyperedges
            for i in range(num_nodes):
                nn_idx = torch.topk(dists_spatial[i], k=k_spatial+1, largest=False).indices
                for node in nn_idx:
                    hyperedge_list.append([node.item() + node_offset, edge_id])
                edge_id += 1

            # Feature hyperedges (based on learned features)
            if k_feature is not None:
                dists_feat = torch.cdist(x_nodes, x_nodes, p=2)
                for i in range(num_nodes):
                    nn_idx = torch.topk(dists_feat[i], k=k_feature+1, largest=False).indices
                    for node in nn_idx:
                        hyperedge_list.append([node.item() + node_offset, edge_id])
                    edge_id += 1

            batch_hyperedge_index.extend(hyperedge_list)
            node_offset += num_nodes

        edge_index = torch.tensor(batch_hyperedge_index, dtype=torch.long, device=x_batched.device).T
        batch_map = torch.cat([torch.full((num_nodes,), b, dtype=torch.long, device=x_batched.device)
                               for b in range(batch_size)])

        return x_batched.view(-1, feat_dim), edge_index, batch_map

class LearnablePatchHyperViG(nn.Module):
    def __init__(self, hidden, num_classes, num_blocks=16, dropout=0.3, k=8):
        super().__init__()
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.k = k

        # removed for now due to crazy slow training rate
        # self.patch_extractor = LearnablePatchExtractor(num_patches=num_patches)

        self.input_proj = nn.Linear(3 * 8 * 8, hidden)
        self.edge_attn = MultiHeadHyperedgeAttention(in_dim=hidden, hidden=64, num_heads=8)

        # Hypergraph blocks
        self.blocks = nn.ModuleList([
            HypergraphBlock(hidden, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, images):
        batch_size = images.size(0)
        device = images.device

        # Extract fixed 8×8 patches
        patch_size = 8
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches_flat = patches.view(batch_size * 16, -1)

        # Build fixed hypergraph
        edge_index, batch_map = self.build_hypergraph(batch_size, device)

        # Project to hidden dimension
        x = self.input_proj(patches_flat)

        # Compute edge attention once
        edge_weight = self.edge_attn(x, edge_index)

        # Process through blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_weight)

        # Pool and classify
        out = self.pool(x, batch_map)
        out = self.classifier(out)

        return out

    def build_hypergraph(self, batch_size, device):
        """Build hypergraph with fixed 4×4 spatial grid"""
        num_patches = 16

        # Fixed spatial coordinates
        coords = torch.tensor([
            [i // 4, i % 4] for i in range(num_patches)
        ], dtype=torch.float, device=device)

        dists = torch.cdist(coords, coords, p=2)

        batch_hyperedge_index = []
        node_offset = 0
        edge_id = 0

        for b in range(batch_size):
            for i in range(num_patches):
                nn_idx = torch.topk(dists[i], k=self.k+1, largest=False).indices
                for node in nn_idx:
                    batch_hyperedge_index.append([node.item() + node_offset, edge_id])
                edge_id += 1
            node_offset += num_patches

        edge_index = torch.tensor(batch_hyperedge_index, dtype=torch.long, device=device).T
        batch_map = torch.cat([
            torch.full((num_patches,), b, dtype=torch.long, device=device)
            for b in range(batch_size)
        ])

        return edge_index, batch_map


class HyperVigClassifier(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, num_blocks=3, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(in_channels, hidden)
        self.edge_attn = MultiHeadHyperedgeAttention(in_dim=hidden, hidden=64, num_heads=8)

        # Multiple hypergraph blocks (each with its own FFN)
        self.blocks = nn.ModuleList([
            HypergraphBlock(hidden, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch_map):
        x = self.input_proj(x)  # [num_nodes, hidden]

        # Compute hyperedge weights in forward pass so gradients flow
        num_hyperedges = edge_index[1].max().item() + 1
        hyperedge_weight = self.edge_attn(x, edge_index)

        # Apply hypergraph blocks (each with its own FFN)
        for block in self.blocks:
            x = block(x, edge_index, hyperedge_weight)

        out = self.pool(x, batch_map)  # [batch_size, hidden]

        out = self.classifier(out)  # [batch_size, num_classes]
        return out


# ============================================================================
# Augmentation Functions
# ============================================================================

def mixup_data(x, y, alpha=0.8):
    """MixUp augmentation: sample-level blending"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation: patch-level blending"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # Get image dimensions (assuming [B, C, H, W])
    _, _, H, W = x.size()
    
    # Generate random bounding box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Clamp bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.8, cutmix_alpha=1.0, mixup_prob=0.5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Apply MixUp/CutMix augmentation
        use_cutmix = False
        if np.random.rand() < mixup_prob:
            if np.random.rand() < 0.5:
                # Apply MixUp
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
            else:
                # Apply CutMix
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
                use_cutmix = True
        else:
            # No augmentation
            labels_a, labels_b = labels, labels
            lam = 1.0
        
        # For LearnablePatchHyperViG, model takes images directly
        if isinstance(model, LearnablePatchHyperViG):
            outputs = model(images)
        else:
            # For other models that use image_to_true_hypergraph
            node_feats, edge_index, batch_map = image_to_true_hypergraph(images)
            outputs = model(node_feats, edge_index, batch_map)
        
        # Compute loss with MixUp/CutMix if applied
        if lam < 1.0:
            if use_cutmix:
                loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if total == 0:
        return None, None
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            if isinstance(model, LearnablePatchHyperViG):
                outputs = model(images)
            else:
                node_feats, edge_index, batch_map = image_to_true_hypergraph(images)
                outputs = model(node_feats, edge_index, batch_map)
            
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Dynamic Hypergraph Edge Attention Model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for CIFAR10 data')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--hidden', type=int, default=320, help='Hidden dimension')
    parser.add_argument('--num_patches', type=int, default=16, help='Number of patches')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help='MixUp alpha parameter')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha parameter')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='Probability of applying MixUp/CutMix')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--neptune_api_key_file', type=str, default='neptune_api_key.txt', 
                       help='Path to file containing Neptune API key')
    parser.add_argument('--neptune_project', type=str, default='31K-ML-Real/381K-Vision-GNN-Project',
                       help='Neptune project name')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save plots')
    
    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # Initialize Neptune monitoring
    neptune_enabled = False
    run = None
    if NEPTUNE_AVAILABLE:
        try:
            if os.path.exists(args.neptune_api_key_file):
                with open(args.neptune_api_key_file, 'r') as f:
                    api_token = f.read().strip()
                run = neptune.init_run(
                    project=args.neptune_project,
                    api_token=api_token,
                )
                # Log hyperparameters
                run["parameters"] = {
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "hidden_dim": args.hidden,
                    "num_patches": args.num_patches,
                    "num_classes": 10,
                    "dropout": args.dropout,
                    "max_epochs": args.epochs,
                    "scheduler_type": "CosineAnnealingLR",
                    "device": device,
                    "model_type": "LearnablePatchHyperViG",
                }
                neptune_enabled = True
                print("Neptune monitoring initialized successfully")
            else:
                print(f"Warning: Neptune API key file '{args.neptune_api_key_file}' not found. Continuing without Neptune monitoring.")
        except Exception as e:
            print(f"Warning: Could not initialize Neptune monitoring: {e}")
            print("Continuing without Neptune monitoring...")
    else:
        print("Neptune not available. Continuing without Neptune monitoring.")

    # Data loading with augmentation
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),  # Random crop with padding
        T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        T.ToTensor(),
        T.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    print("Loading CIFAR10 dataset...")
    train_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Model
    print("Initializing model...")
    # Slightly reduce model capacity to reduce overfitting
    model = LearnablePatchHyperViG(
        hidden=args.hidden,
        num_classes=10,
        num_blocks=5,  # Reduced from 6 to 5
        dropout=args.dropout,
        k=8
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Label smoothing loss
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop with early stopping
    best_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob
        )
        
        # Check if training returned None (NaN detected)
        if train_loss is None or train_acc is None:
            print(f"Training stopped due to NaN/Inf at epoch {epoch+1}")
            break
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Check for NaN in validation
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"NaN detected at epoch {epoch+1}! Training stopped.")
            break
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Log metrics to Neptune
        if neptune_enabled and run is not None:
            try:
                run["train/loss"].append(train_loss)
                run["train/accuracy"].append(train_acc)
                run["val/loss"].append(val_loss)
                run["val/accuracy"].append(val_acc)
                run["learning_rate"].append(current_lr)
            except Exception as e:
                print(f"Warning: Could not log to Neptune: {e}")
        
        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model! Saving to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"\nEarly stopping triggered! No improvement for {args.early_stop_patience} epochs.")
                print(f"Best validation accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")
                break
        
        # Save periodic checkpoints
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc*100:.2f}%")
    
    # Create and save plots
    if len(train_losses) > 0 and len(val_losses) > 0:
        try:
            plt.figure(figsize=(12, 5))

            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o', markersize=3)
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            # Plot accuracy
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', marker='o', markersize=3)
            plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plot_path = os.path.join(args.plot_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nTraining curves saved to '{plot_path}'")
            
            # Upload plot to Neptune
            if neptune_enabled and run is not None:
                try:
                    run["training_curves"].upload(plot_path)
                    print("Training curves uploaded to Neptune")
                except Exception as e:
                    print(f"Warning: Could not upload plot to Neptune: {e}")
            
            plt.close()  # Close the figure to free memory
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    else:
        print("Warning: No training data collected, skipping plot generation")

    # Stop Neptune run
    if neptune_enabled and run is not None:
        try:
            run.stop()
            print("Neptune run stopped successfully")
        except Exception as e:
            print(f"Warning: Could not stop Neptune run: {e}")


if __name__ == '__main__':
    main()
