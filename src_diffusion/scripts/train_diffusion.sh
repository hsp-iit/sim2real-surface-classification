CKPT_ROOT="/mnt/hpc_data/shared_sim2real/ckpts/diffusion/"
ROOT_DATASETS="/mnt/hpc_data/shared_sim2real/datasets/"

#########################################################################################

OUT_PATH=${CKPT_ROOT}sim2real_nobg_pre/
CONFIG_PATH=${OUT_PATH}train_diffusion.yaml
DATA_PATH=${ROOT_DATASETS}real_dataset/nobg/

# Uncomment to train conditioned diffusion
#CONDITION_FOLDER=${ROOT_DATASETS}masks/mask0/

#########################################################################################
# training info
GPUS=4
BATCH_SIZE=16
GRAD_ACC=4
MIXED_PRECISION="no"
GPU_IDS="all"

if (( $GPUS > 1 )) ; then
    MULTI="--multi_gpu"
else
    MULTI=""
fi

accelerate launch \
    --gpu_ids=$GPU_IDS \
    --num_processes $GPUS \
    --mixed_precision $MIXED_PRECISION \
    $MULTI \
train_diffusion.py \
    --dataset_path $DATA_PATH \
    --out_path $OUT_PATH \
    --config_path $CONFIG_PATH \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulate_every $GRAD_ACC \
    #--condition_folder $CONDITION_FOLDER 
