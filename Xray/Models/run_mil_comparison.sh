#!/bin/bash
#SBATCH --job-name=MIL_models_comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=14:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "Running on host: $HOSTNAME"
echo "-------------------------------"
start=$(date +%s)

# Setup environment
module load anaconda3/2023.09-0
module load cuda/12.3.0
source activate pytorch4d  # Replace with your actual environment name

# Change to your working directory
cd /home/xiaofey/xray/xray-master/code  # Replace with your actual directory

# Run MIL scripts
for model in MIL1 MIL2 MIL3 MIL4
do
    echo "Running ${model}.py..."
    python ${model}.py
done

# Run combining script
echo "Combining results..."
python combine_boxplots.py 
end=$(date +%s)
runtime=$((end-start))
echo "-------------------------------"
echo "Total runtime (s): $runtime"
echo "-------------------------------"



