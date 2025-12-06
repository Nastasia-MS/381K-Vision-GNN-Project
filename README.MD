# 381K Vision GNN Project

A research project exploring Vision Graph Neural Networks (GNNs) using dynamic hypergraphs for image classification. This project implements progressive improvements from basic image-to-graph conversion to advanced hypergraph edge attention mechanisms.

## Project Overview

This project implements a series of Vision GNN models that convert images into graph structures and process them using hypergraph neural networks. The work progresses through multiple versions (V0-V4), each building upon the previous iteration with enhanced capabilities.

## Version Progression

The project follows a linear progression through five versions, organized into three milestones:

### Milestone 1: Foundation (V0 - V2)
- **V0 - Image to Graph**: Initial implementation focusing on converting images to graph structures
- **V1 - Dynamic Hypergraph**: Introduction of dynamic hypergraph construction
- **V2 - Hypergraph VIG**: Integration with Vision GNN (ViG) architecture

### Milestone 2: Edge Attention (V3)
- **V3 - Dynamic Hypergraph Edge Attention**: Implementation of edge attention mechanisms in hypergraph neural networks

### Milestone 3: Advanced Training (V4)
- **V4 - Dynamic Hypergraph Edge Attention (Enhanced)**: Final version with improved training procedures, data augmentation, and monitoring capabilities

Each version is available both as a Jupyter notebook (for exploration and development) and as a Python script (for training on TACC).

## Project Structure

```
381K-Vision-GNN-Project/
├── data/                          # Dataset directory
│   └── cifar-10-batches-py/      # CIFAR-10 dataset
├── Vision_GNN/                    # Base Vision GNN library (from external repo)
│   ├── gcn_lib/                   # Graph convolution library
│   ├── data/                      # Data loaders and utilities
│   └── ...
├── Figures/                       # Generated visualization figures
├── Output/                        # Training output logs
├── Error/                         # Error logs from TACC jobs
├── V0-Image_to_Graph.ipynb        # Milestone 1: Image to graph conversion
├── V1-Dynamic_Hypergraph.ipynb    # Milestone 1: Dynamic hypergraph
├── V2-Hypergraph_VIG.ipynb        # Milestone 1: Hypergraph VIG
├── V3-Dynamic_Hypergraph_Edge_Attention.ipynb  # Milestone 2: Edge attention
├── V4-Dynamic_Hypergraph_Edge_Attention.ipynb  # Milestone 3: Enhanced version
├── v1-dynamic_hypergraph_edge_attention.py     # Python script for V1
├── v3-dynamic_hypergraph_edge_attention.py     # Python script for V3
├── v4-dynamic_hypergraph_edge_attention.py     # Python script for V4
├── run_dynamic_hypergraph.sh      # SLURM script for TACC training
├── requirements.txt               # Python dependencies
├── neptune_api_key.txt            # Neptune.ai API key (keep private!)
└── README.md                      # This file
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for training)
- Access to TACC (Texas Advanced Computing Center) for distributed training
- Neptune.ai account (for experiment tracking)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 381K-Vision-GNN-Project
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `torch-geometric>=2.3.0` - Graph neural network library
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Visualization
- `neptune>=1.0.0` - Experiment tracking

### 3. Setup Neptune.ai Monitoring

1. Create a Neptune.ai account at [https://neptune.ai](https://neptune.ai)
2. Create a new project (or use existing): `31K-ML-Real/381K-Vision-GNN-Project`
3. Get your API token from the Neptune dashboard
4. Save your API token to `neptune_api_key.txt`:
   ```bash
   echo "your-api-token-here" > neptune_api_key.txt
   ```

**Note:** Keep `neptune_api_key.txt` private and do not commit it to version control. Add it to `.gitignore` if needed.

## Dataset Setup

The project uses the CIFAR-10 and CIFAR-100 datasets. The datasets will be automatically downloaded when you run the training scripts for the first time.

### Automatic Download

The Python scripts automatically download the datasets to the `./data` directory:

```python
from torchvision.datasets import CIFAR10, CIFAR100

# CIFAR-10 (used in V0-V3)
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# CIFAR-100 (used in V4)
train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
```

### Manual Download (Optional)

If you prefer to download manually:
1. Download CIFAR-10: [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Download CIFAR-100: [https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
3. Extract to `./data/` directory

## Training on TACC

This project is designed to run on TACC (Texas Advanced Computing Center) using SLURM job scheduling.

### 1. SSH Connection to TACC

Connect to TACC using SSH:

```bash
ssh <your-username>@stampede2.tacc.utexas.edu
# or
ssh <your-username>@frontera.tacc.utexas.edu
```

### 2. Setup Environment on TACC

Once connected, set up your environment:

```bash
# Navigate to your project directory
cd /path/to/381K-Vision-GNN-Project

# Load required modules
module reset
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8

# Activate virtual environment (if using one)
source venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 3. Running Training Jobs

Training is done using SLURM batch scripts. The main script is `run_dynamic_hypergraph.sh`.

#### Submit a Training Job

```bash
sbatch run_dynamic_hypergraph.sh
```

#### Check Job Status

```bash
# Check your jobs in the queue
squeue -u <your-username>

# View detailed job information
scontrol show job <jobID>
```

#### Monitor Job Output

- **Standard output**: Check `Output/out_<jobID>.txt`
- **Error logs**: Check `Error/error_<jobID>.txt`

#### Cancel a Job

```bash
scancel <jobID>
```

### 4. Interactive GPU Node (for Testing)

For interactive testing and debugging, you can request an interactive GPU node:

```bash
idev -p gh-dev -t 2:00:00
```

This gives you a dedicated interactive GPU node for 2 hours, useful for:
- Testing code before submitting batch jobs
- Debugging issues
- Running quick experiments

### 5. Training Script Configuration

The `run_dynamic_hypergraph.sh` script is configured for:
- **Partition**: `gg` (Grace Hopper GPU partition)
- **Nodes**: 1
- **Time limit**: 24 hours
- **Output/Error**: Saved to `Output/` and `Error/` directories

You can modify the script to:
- Change the Python file being executed (currently `v4-dynamic_hypergraph_edge_attention.py`)
- Adjust resource requirements
- Modify email notifications

**Example modification for different versions:**

```bash
# For V3
python v3-dynamic_hypergraph_edge_attention.py

# For V4 (current)
python v4-dynamic_hypergraph_edge_attention.py
```

## Local Development

For local development and experimentation, use the Jupyter notebooks:

1. **V0-Image_to_Graph.ipynb**: Start here to understand image-to-graph conversion
2. **V1-Dynamic_Hypergraph.ipynb**: Explore dynamic hypergraph construction
3. **V2-Hypergraph_VIG.ipynb**: Learn about Hypergraph VIG integration
4. **V3-Dynamic_Hypergraph_Edge_Attention.ipynb**: Study edge attention mechanisms
5. **V4-Dynamic_Hypergraph_Edge_Attention.ipynb**: Review the final enhanced version

## Experiment Monitoring with Neptune.ai

The project integrates with Neptune.ai for experiment tracking and monitoring.

### Features Tracked

- **Hyperparameters**: Learning rate, batch size, model architecture, etc.
- **Training Metrics**: Loss, accuracy per epoch
- **Validation Metrics**: Validation loss and accuracy
- **Training Curves**: Automatically generated and uploaded plots
- **System Information**: GPU details, CUDA version, device information

### Viewing Results

1. Log in to [https://app.neptune.ai](https://app.neptune.ai)
2. Navigate to your project: `31K-ML-Real/381K-Vision-GNN-Project`
3. View all runs, compare experiments, and analyze results

### Disabling Neptune (Optional)

If you want to run without Neptune monitoring, the scripts will continue to work. Simply ensure `neptune_api_key.txt` doesn't exist or remove the Neptune import. The code gracefully handles missing Neptune dependencies.

## Usage Examples

### Running V4 Training Locally

```bash
python v4-dynamic_hypergraph_edge_attention.py \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden 256 \
    --data_dir ./data
```

### Running with Custom Parameters

```bash
python v4-dynamic_hypergraph_edge_attention.py \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.0005 \
    --hidden 512 \
    --dropout 0.3 \
    --weight_decay 0.0001 \
    --neptune_project "31K-ML-Real/381K-Vision-GNN-Project"
```

## Key Features

### V4 Model Highlights

- **Learnable Patch Hypergraph ViG**: Advanced architecture with learnable patch embeddings
- **Dynamic Hypergraph Construction**: Adaptive hypergraph building based on spatial and feature similarities
- **Edge Attention Mechanisms**: Attention-based aggregation for hypergraph edges
- **Data Augmentation**: Random crop, horizontal flip, and normalization
- **CIFAR-100 Support**: Extended to 100-class classification
- **Comprehensive Monitoring**: Integrated Neptune.ai tracking

## Output Files

- **Model Checkpoints**: Saved to `./checkpoints/` (if enabled)
- **Training Plots**: Saved to `./plots/` directory
- **Logs**: Training logs and metrics
- **Neptune Dashboard**: Online experiment tracking

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in the training script
2. **Neptune Connection Errors**: Check API key and internet connection
3. **Dataset Download Issues**: Ensure internet connection and sufficient disk space
4. **TACC Job Failures**: Check error logs in `Error/error_<jobID>.txt`

### Getting Help

- Check error logs in the `Error/` directory
- Review output logs in the `Output/` directory
- Consult TACC documentation for SLURM issues
- Review Neptune.ai dashboard for training metrics

## References

This project builds upon and is inspired by the following research works:

```bibtex
@Article{han_wang_2022vision_cnn,
  author  = {Kai Han and Yunhe Wang and others},
  title   = {Vision GNN: An Image is Worth Graph of Nodes},
  year    = {2022}
}

@article{munir_avery2023mobilevig,
  author    = {Mustafa Munir and William Avery and others},
  title     = {MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications},
  year      = {2023}
}

@Article{park_kim2022vit,
  author  = {Namuk Park and Songkuk Kim},
  title   = {How Do Vision Transformers Work?},
  year    = {2022}
}

@article{krizhevsky2009cifar,
  author      = {Krizhevsky, A.},
  title       = {Learning Multiple Layers of Features from Tiny Images},
  institution = {University of Toronto},
  year        = {2009}
}

@Article{fey2019pytorch,
  author  = {Fey, M. and Lenssen, J. E.},
  title   = {PyTorch Geometric: Deep Learning on Graphs with PyTorch},
  year    = {2019}
}

@Article{gedik2025attentionvig,
  author  = {Hakan Emre Gedik and Andrew Martin and Mustafa Munir and others},
  title   = {AttentionViG: Cross-Attention-Based Dynamic Neighbor Aggregation in Vision GNNs},
  year    = {2025}
}

@Article{munir2024greedyvig,
  author = {Munir Mustafa and Avery William and others},
  title  = {GreedyViG: Dynamic Axial Graph Construction for Efficient Vision GNNs},
  year   = {2024}
}

@Article{parikh2025clustervig,
  title  = {ClusterViG: Efficient Globally Aware Vision GNNs via Image Partitioning},
  author = {Parikh Dhruv and Fein-Ashley Jacob and others},
  year   = {2025}
}

@Article{fixelle2025hgvt,
  title   = {Hypergraph Vision Transformers: Images are More than Nodes, More than Edges},
  author  = {Fixelle, Joshua},
  year    = {2025}
}
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vision-gnn-project,
  title={381K Vision GNN Project: Dynamic Hypergraph Edge Attention},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```


## Acknowledgments
- Radu Marculescu: Professor and the Laura Jennings Turner Chair in Engineering in the Department of Electrical and Computer Engineering at The University of Texas at Austin
- Mustafa Munir: Ph.D. Student, Efficient Computer Vision, Generative AI

- Vision GNN library: [https://github.com/jichengyuan/Vision_GNN](https://github.com/jichengyuan/Vision_GNN)
- TACC for computational resources
- Neptune.ai for experiment tracking

## Contact

For questions or issues, please contact: [laansalurasmus@gmail.com,n.maldeistumm@utexas.edu]
