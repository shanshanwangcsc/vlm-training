#!/bin/bash
#SBATCH --account=project_462001202
#SBATCH --partition=dev-g
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --output=logs/%j/log_%x.out
#SBATCH --error=logs/%j/errors/rank_%t.err
#SBATCH --exclude=nid005002,nid005019

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


# =============================================================================
# Environment configuration
# =============================================================================
# Locale
export LC_ALL=C
export LANG=C
export PYTHONNOUSERSITE=1

# Cray MPICH Configuration
export MPICH_GPU_SUPPORT_ENABLED=1          # Enable GPU-aware MPI
export MPICH_OFI_NIC_POLICY="GPU"           # Use GPU-mapped NICs
export MPICH_RANK_REORDER_METHOD=1          # Optimize rank placement
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

# RCCL Configuration
export RCCL_DEBUG="${RCCL_DEBUG:-INFO}"
export NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"
export NCCL_NET_GDR_LEVEL="PHB"

# Libfabric/CXI for Slingshot
export FI_CXI_ATS=0
export FI_CXI_DISABLE_CQ_HUGETLB=1
export FI_MR_CACHE_MONITOR=userfaultfd

# Set up the CPU bind masks
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

# srun should not apply a restrictive cpu mask because torchrun spawns 8 processes.
# The python code will handle cpu affinity via psutil.
srun --cpu-bind=none singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "python -m torch.distributed.run \
--nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE \
--rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
-m train.train_qwen --config configs/lumi/qwen3_5_9b.toml"

#srun singularity run --env PYTHONPATH=$PYTHONPATH:\$PYTHONPATH ${SIF} bash -c "python -m torch.distributed.run \
#--nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_PER_NODE \
#--rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
#-m train.train_qwen --config configs/lumi/qwen3_2b.toml"