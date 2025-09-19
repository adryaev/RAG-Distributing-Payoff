#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100_80gb:1      # Only use the A100 GPUs with 80GB
#SBATCH --cpus-per-gpu=8
##SBATCH --mem=64G                    # Request 64GB of RAM
#SBATCH --time=13:00:00            # Maximum amount of time the job will run: 2 hours, 30 minutes, and 2 seconds
#SBATCH --job-name="RAG payoff"    # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=outputs/jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=outputs/jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=alan.dryaev@stud.uni-hannover.de # Email address to send the email to

module load conda
conda activate rag

python3 rag.py
