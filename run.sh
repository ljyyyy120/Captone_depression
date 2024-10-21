#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8GB
#SBATCH --time=1:00:00

# Load Anaconda module
module purge;
module load anaconda3/2020.07;

# Set the number of OpenMP threads to match SLURM CPU settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

# Activate the conda shell script to allow conda commands
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

# If the environment doesn't exist, create it from the .yml file
if [ ! -d "./penv" ]; then
  conda env create -f environment.yml -p ./penv
fi

# Activate the environment
conda activate ./penv;

# Add the environment's bin directory to the PATH
export PATH=./penv/bin:$PATH;

# Ensure the additional package is installed if needed
# conda install <your_package_name>  # Uncomment this line to install any additional packages

# Run the Python script
python Depression_Detector.py