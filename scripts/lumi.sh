#!/bin/bash
#SBATCH --account=project_462001202
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --output=logs/%j/log_%x.out
#SBATCH --error=logs/%j/errors/rank_%t.err

# ---------- Modules ----------
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# set MIOPEN temp folder
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config


export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export DOMAIN_BLACKLIST=github.com,huggingface.co

# ---------- Container ----------
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
export PYTHONPATH="/scratch/project_462001202/shanshan/vlm-env/lib/python3.12/site-packages:$PYTHONPATH"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}" # set port based on SLURM_JOB_ID to avoid conflicts

# To have RCCL use the Slingshot interfaces:
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# To have RCCL use GPU RDMA:
export NCCL_NET_GDR_LEVEL=PHB

#srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "python -m torch.distributed.run \
#--nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE \
#--master_addr=$MASTER_ADDR \
#--master_port=$MASTER_PORT \
#-m train.train_qwen --config configs/lumi/qwen3_2b.toml"

srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "python -m torch.distributed.run \
--nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
-m train.train_qwen --config configs/lumi/qwen3_2b.toml"