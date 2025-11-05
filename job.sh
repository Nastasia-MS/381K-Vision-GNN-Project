#!/bin/bash

#SBATCH -J dynamic_hypergraph_test
#SBATCH -o Output/out_%j.txt
#SBATCH -e Error/error_%j.txt
#SBATCH -p gg                                              # Grace Hopper GPU partition
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:01:00
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
python test.py

echo "=========================================="
echo "End Time: $(date)"
echo "Test completed successfully!"
echo "=========================================="
