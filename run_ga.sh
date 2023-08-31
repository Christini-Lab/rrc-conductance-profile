#! /bin/bash -l

#SBATCH --partition=panda
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --job-name=GA-1
#SBATCH --time=120:00:00             #HH/MM/SS
#SBATCH --mem=150G                    #memory requested, units available K,M,G,T

#source ~/ .bashrc

export save_name=$1 #'./data/trial1_'
echo "Starting at:" `date` >> ga_output.txt
sleep 30
echo "This is job #:" $SLURM_JOB_ID >> ga_output.txt
echo "Running on node:" `hostname` >> ga_output.txt
echo "Running on cluster:" $SLURM_CLUSTER_NAME >> ga_output.txt
echo "This job was assigned the temporary (local) directory:" $TMPDIR >> ga_output.txt


module load python-3.7.6-gcc-8.2.0-hk56qj4
module load sundials/5.7.0
python3 run_ga.py