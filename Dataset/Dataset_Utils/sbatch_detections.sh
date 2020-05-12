#!/bin/bash
#SBATCH --time=22:0:0
#SBATCH -o "AFFtrainrec.extraction.%j.log"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=120GB


module load Python/3.6.4-fosscuda-2018a
module load FFmpeg/4.1.3-GCCcore-8.2.0
module load CMake/3.13.3-GCCcore-8.2.0 
module load CUDA/10.0.130
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/s4179447

python3 generate_detections.py