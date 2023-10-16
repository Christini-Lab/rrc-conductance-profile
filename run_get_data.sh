#! /bin/bash -l

#SBATCH --partition=panda
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sen-t
#SBATCH --time=24:00:00             #HH/MM/SS
#SBATCH --mem=50G                  #memory requested, units available K,M,G,T

#source ~/ .bashrc

echo "Starting at:" `date` >> run_get_data.txt
sleep 30
echo "This is job #:" $SLURM_JOB_ID >> run_get_data.txt
echo "Running on node:" `hostname` >> run_get_data.txt
echo "Running on cluster:" $SLURM_CLUSTER_NAME >> run_get_data.txt
echo "This job was assigned the temporary (local) directory:" $TMPDIR >> run_get_data.txt


module load python-3.7.6-gcc-8.2.0-hk56qj4
module load sundials/5.7.0
python3 get_data.py 