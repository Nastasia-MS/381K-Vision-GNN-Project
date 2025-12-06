#!/bin/bash

#SBATCH -J dynamic_hypergraph_cifar100
#SBATCH -o Output/out_%j.txt
#SBATCH -e Error/error_%j.txt
#SBATCH -p gh                                              # Grace Hopper GPU partition
#SBATCH -N 1                                               # Use 1 node (required for GPU allocation)
#SBATCH -n 1                                               # Use 1 task                                 # Request 1 GPU per node (alternative syntax)
#SBATCH -t 00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user='rl37272@my.utexas.edu'

module reset
module load gcc/13.2.0
module load cuda/12.8
module load python3/3.11.8
source venv/bin/activate

# ==============================
# 3️⃣ Log metadata
# ==============================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# ==============================
# 5️⃣ Run test script
# ==============================
python --version
which python
python v4-dynamic_hypergraph_edge_attention.py

echo "=========================================="
echo "End Time: $(date)"
echo "Test completed successfully!"
echo "=========================================="
