# RAG-Distributing-Payoff


## 0. Connect to ssh neumann cluster

## 1. Create Conda Environment

First, create a dedicated Conda environment. This project requires Python 3.10.
Bash

conda create -n rag python=3.10.17
conda activate rag

## 2. Install Dependencies

Install all the required Python packages using the requirements.txt file.
Bash

pip install -r requirements.txt

## 3. Pyserini & MS MARCO Indexes

rag.py requires msmarco-v2-passage index.

evaluate.py requires msmarco-v2-passage, msmarco-v2-doc index.


Note: Saved by default to ~/.cache/pyserini/indexes/


## 4. Running RAG Pipeline

main retrieval and generation process

python3 rag.py

Progress Tracking: progress number is saved in progress.json. If the script is interrupted, it can resume from where it left off.

Output: The final results are saved in results.jsonl. Each line in this file corresponds to a query, the generated output, and the retrieved passage ids.



## 5. run python3 rag.py

for testing:
module load conda
conda activate rag
srun --gres=gpu:a100_80gb:1 --cpus-per-gpu=8 --mem=64G  -t 30:00 --pty /bin/bash -i

slurm job:
sbatch rag.sh

progess is saved in progress.json

output: results.jsonl


# a. Interactive Session

To request an interactive session on a GPU node for testing or debugging:

module load conda
conda activate rag
srun --gres=gpu:a100_80gb:1 --cpus-per-gpu=8 --mem=64G -t 00:30:00 --pty /bin/bash -i

# b. Submitting a Batch Job

sbatch rag.sh


# Output: 
dcm_counts.csv: A list of domains and their citation counts. Every citation is counted, so if a domain is cited multiple times for a single query, each instance is counted.

dcs_counts.csv: A list of domains and their citation counts. A domain is counted only once per query, even if multiple documents from that same domain are cited.

domain_docids_multiple.jsonl: A dictionary mapping each domain to a list of the MS MARCO docids cited from it (follows the "multiple" counting logic).

domain_docids_single.jsonl: A dictionary mapping each domain to a list of the MS MARCO docids cited from it (follows the "single" counting logic).

# 6. Create Graph with graph.ipynb


# 7. Future Usage

In the evaluate.py you can customize the weights by passing the weighting_function as parameter to count_domains function.

Optionally you can analyze the domain_docids_single/multiple manually.