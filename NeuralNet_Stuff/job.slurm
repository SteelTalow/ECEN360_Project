#!/bin/bash
#SBATCH --job-name=JobExample4       #Set the job name to "JobExample4"
#SBATCH --time=00:10:00              #Set the wall clock limit to 10
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=2560M                  #Request 2560MB (2.5GB) per task
#SBATCH --output=Example4Out.%j      #Send stdout/err to "Example4Out.[jobID]"

module restore d2

source $SCRATCH/myENV/bin/activate

cd $SCRATCH

srun python neuralNET.py