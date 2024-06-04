#!/bin/bash
#SBATCH --job-name=test_finetune_poro
#SBATCH --account=project_462000558
#SBATCH --partition=dev-g
#SBATCH --cpus-per-task=56 #Tune this to your needs. 56 cpu cores is the maximum for 1 node
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --exclusive    #Make sure you have all of the node's resources
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

# auto-fail on any errors in this script
set -eox pipefail

echo "PWD" $PWD #Sanity check

module use /appl/local/csc/modulefiles/ #On Lumi the paths for the csc provided modules arent provided by default
module load pytorch


export HF_HOME=/scratch/project_462000558/cache

#Variables for distributed enviroment
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr #IP adress of the first node
export LOCAL_RANK=$SLURM_LOCALID #Rank of the processes inside 1 distributed group
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES)) #Overall amount of processes/gpus

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 #Reduce verbosity in logs
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false #Removes error involved with the FastTokenizer and rust/python parallelism.
                                    #See more:https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996

#Accelerate config for distributed training
ACCELERATE_CONFIG_FILE=configs/deepspeed_zero3.yaml

echo "JOBNAME" $SLURM_JOB_NAME
echo "CONFIG" $ACCELERATE_CONFIG_FILE

export CMD=" \
    finetune.py --model LumiOpen/Poro-34B"


#LAUNCHER
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

srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$ACC_LAUNCHER --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

echo "END $SLURM_JOBID: $(date)"