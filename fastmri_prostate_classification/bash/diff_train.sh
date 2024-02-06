#!/bin/bash
#SBATCH -p gpu4_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=diff
#SBATCH --output=fastMRI_prostate_dwi_%j.log
#SBATCH --array=0


echo "Setting up environment"
module load anaconda3/gpu/5.2.0
module load cuda/10.1.105
conda activate /gpfs/data/brownrlab/20210914_conda
export PYTHONPATH=/gpfs/data/brownrlab/20210914_conda/lib/python3.9/site-packages
export LD_LIBRARY_PATH="/gpfs/data/brownrlab/20210914_conda/lib:$LD_LIBRARY_PATH" 
echo "completed environment set up"

echo "started training"
python -u ../main_diff.py \
    --config_file ../configs/diffusion_final.yaml \
    --index_seed ${SLURM_ARRAY_TASK_ID}
echo "completed training" 
