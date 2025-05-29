#!/bin/sh
#SBATCH --job-name=KW_RL
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00 
#SBATCH -c 1
#SBATCH --output=/n/home00/akarchmer/kw-rl/%A__%a.out
#SBATCH --mail-user=akarchmer0@gmail.com
#SBATCH --mail-type=ALL

# 1 Load required modules
module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01

# 2 Activate your env
eval "$(mamba shell.bash hook)"
mamba activate kw_rl_env

# 3 Install Python dependencies
#    (only takes effect if there are updates/new installs)
cd /n/home00/akarchmer/kw-rl
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# 4 Set output directory
OUT=/n/home00/akarchmer/kw-rl/results
mkdir -p "$OUT"

# 5 Run the training script
python KW_RL.py > "$OUT"/training_$(date +%Y%m%d_%H%M%S).log
