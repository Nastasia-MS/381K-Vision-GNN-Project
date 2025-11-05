#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to verify all required libraries are working
Tests imports and basic functionality of libraries used in dynamic_hypergraph_edge_attention.py
"""

import sys
import os

print("=" * 60)
print("Testing Library Imports and Functionality")
print("=" * 60)

# Test numpy
print("\n[1/8] Testing numpy...")
try:
    import numpy as np
    arr = np.array([1, 2, 3])
    print(f"  ✓ numpy imported successfully (version: {np.__version__})")
    print(f"  ✓ numpy test array: {arr}")
except Exception as e:
    print(f"  ✗ numpy failed: {e}")
    sys.exit(1)

# Test torch
print("\n[2/8] Testing torch...")
try:
    import torch
    print(f"  ✓ torch imported successfully (version: {torch.__version__})")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
    x = torch.randn(2, 3)
    print(f"  ✓ torch test tensor shape: {x.shape}")
except Exception as e:
    print(f"  ✗ torch failed: {e}")
    sys.exit(1)

# Test torch.nn
print("\n[3/8] Testing torch.nn...")
try:
    import torch.nn as nn
    linear = nn.Linear(10, 5)
    print(f"  ✓ torch.nn imported successfully")
    print(f"  ✓ Created Linear layer: {linear}")
except Exception as e:
    print(f"  ✗ torch.nn failed: {e}")
    sys.exit(1)

# Test torchvision.transforms
print("\n[4/8] Testing torchvision.transforms...")
try:
    import torchvision.transforms as T
    transform = T.ToTensor()
    print(f"  ✓ torchvision.transforms imported successfully")
    print(f"  ✓ Created ToTensor transform: {transform}")
except Exception as e:
    print(f"  ✗ torchvision.transforms failed: {e}")
    sys.exit(1)

# Test torch.nn.functional
print("\n[5/8] Testing torch.nn.functional...")
try:
    import torch.nn.functional as F
    x = torch.randn(2, 3)
    relu_x = F.relu(x)
    print(f"  ✓ torch.nn.functional imported successfully")
    print(f"  ✓ ReLU test: input shape {x.shape}, output shape {relu_x.shape}")
except Exception as e:
    print(f"  ✗ torch.nn.functional failed: {e}")
    sys.exit(1)

# Test torchvision.datasets
print("\n[6/8] Testing torchvision.datasets...")
try:
    from torchvision.datasets import CIFAR10
    print(f"  ✓ torchvision.datasets imported successfully")
    print(f"  ✓ CIFAR10 dataset class available")
except Exception as e:
    print(f"  ✗ torchvision.datasets failed: {e}")
    sys.exit(1)

# Test torch.utils.data
print("\n[7/8] Testing torch.utils.data...")
try:
    from torch.utils.data import DataLoader
    print(f"  ✓ torch.utils.data imported successfully")
    print(f"  ✓ DataLoader class available")
except Exception as e:
    print(f"  ✗ torch.utils.data failed: {e}")
    sys.exit(1)

# Test torch_geometric
print("\n[8/8] Testing torch_geometric...")
try:
    from torch_geometric.nn import HypergraphConv, AttentionalAggregation
    print(f"  ✓ torch_geometric imported successfully")
    print(f"  ✓ HypergraphConv available: {HypergraphConv}")
    print(f"  ✓ AttentionalAggregation available: {AttentionalAggregation}")
    
    # Test creating a simple hypergraph conv layer
    conv = HypergraphConv(64, 32)
    print(f"  ✓ Created HypergraphConv layer: {conv}")
except Exception as e:
    print(f"  ✗ torch_geometric failed: {e}")
    sys.exit(1)

# Test matplotlib
print("\n[9/9] Testing matplotlib...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print(f"  ✓ matplotlib imported successfully")
    print(f"  ✓ matplotlib backend: {matplotlib.get_backend()}")
    
    # Test creating a simple plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], [1, 4, 2])
    test_plot_path = 'test_plot.png'
    plt.savefig(test_plot_path)
    plt.close()
    if os.path.exists(test_plot_path):
        print(f"  ✓ Successfully created test plot: {test_plot_path}")
        os.remove(test_plot_path)  # Clean up
except Exception as e:
    print(f"  ✗ matplotlib failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nAll required libraries are working correctly.")
print("You can now run dynamic_hypergraph_edge_attention.py")
print("=" * 60)

