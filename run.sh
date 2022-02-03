#!/bin/bash
#SBATCH --job-name=cyto_folding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=96:00:00
#SBATCH --output=/home/mvries/FoldingNetNew/outputs/%joutput.out
#SBATCH --error=/home/mvries/FoldingNetNew/error_file/%jerror.err
#SBATCH --partition=gpu
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate dcfn


python main.py --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/FoldingNetNew/' --dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/Single_cell_Treat_PointCloud/' --dataframe_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/cleanedDataNew.csv' --num_features 50 --shape 'sphere' 
