#!/bin/bash
#SBATCH --job-name=MIL_models_comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
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

# Define the range of attention branches to test
BRANCHES=(1 2 4 8 16 32 64 128 256 512 1024 2048)

for branches in "${BRANCHES[@]}"
do
    echo "Running Attn_head with ${branches} attention branches..."
    python Attn_head.py --save_dir "/home/xiaofey/xray/xray-master/code/saved_models_MIL4_att_test" \
                   --plot_dir "/home/xiaofey/xray/xray-master/code/plots_MIL4_att_test" \
                   --pretrained_model_dir "/home/xiaofey/xray/xray-master/code/test6" \
                   --attention_branches ${branches}
    

    # if [ $? -ne 0 ]; then
    #     echo "Error occurred while running MIL4.py with ${branches} attention branches"
    #     exit 1
    # fi
done

end=$(date +%s)
runtime=$((end-start))
echo "-------------------------------"
echo "Total runtime (s): $runtime"
echo "-------------------------------"