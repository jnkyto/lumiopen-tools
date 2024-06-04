#!/bin/bash
#SBATCH --job-name=ft_tr_poro
#SBATCH --account=project_462000558
#SBATCH --time=00:10:00
#SBATCH --partition=dev-g
#SBATCH --mem=480G
#SBATCH --nodes=3
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --error=slurm_out/slurm_%j.err

# Create log directory if it doesn't exist
mkdir -p slurm_out
# Remove previously created symlinks just in case
rm -f slurm_out/latest.out slurm_out/latest.err
# Create latest log symlinks for current job
ln -s slurm_$SLURM_JOB_ID.out slurm_out/latest.out
ln -s slurm_$SLURM_JOB_ID.err slurm_out/latest.err

# Autofail on sbatch errors
set -eox pipefail

# Logging script's variables/commands for future debug needs
set -x

# Sanity check
echo "PWD:" $PWD

module use /appl/local/csc/modulefiles
module load pytorch
export HF_HOME=/scratch/project_462000558/cache

# Variables for distributed enviroment
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr #IP adress of the first node
export LOCAL_RANK=$SLURM_LOCALID #Rank of the processes inside 1 distributed group
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES)) #Overall amount of processes/gpus

export TRANSFORMERS_NO_ADVISORY_WARNINGS=0 # Toggle to reduce verbosity
export LOG_LEVEL=INFO
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false #Removes error involved with the FastTokenizer and rust/python parallelism.
                                    #See more:https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996

# Accelerate config for distributed training
ACCELERATE_CONFIG_FILE=config/deepspeed_zero3.yaml

echo "JOBNAME:" $SLURM_JOB_NAME
echo "CONFIG:" $ACCELERATE_CONFIG_FILE

###########################################
# THIS IS PROBABLY WHAT YOU WANT TO TWEAK #
###########################################
export CMD=" \
    main_ct.py -b 1 -e 1"

# Accelerate launcher
export ACC_LAUNCHER="singularity_wrapper exec accelerate launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --tee 3 \
    "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS --jobid $SLURM_JOBID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"
echo "END $SLURM_JOBID: $(date)"