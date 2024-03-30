N_CLASSES=4

# DATASET TRAIN SOURCE
DATA_SRC="/mnt/hpc_data/shared_sim2real/datasets/databases/train_dataset_balanced/"
LABELS="labels.csv"

# DATASET TRAIN TARGET
DATA_TGT="/mnt/hpc_data/shared_sim2real/datasets/real_dataset/nobg/"

# DATASET TEST
DATA_TEST="/mnt/hpc_data/shared_sim2real/datasets/databases/final_testset/"

# CONFIG
CONFIG="config/dann_hparams.yaml"
CONFIG_MODEL="config/dino_v2_train_config.yaml"


#########################################################################################
# training info
GPUS=4
BATCH_SIZE=64 #128
GRAD_ACC=1
MIXED_PRECISION="no"
GPU_IDS="all"


if (( $GPUS > 1 )) ; then
    MULTI="--multi_gpu"
else
    MULTI=""
fi

echo "STARTING FINETUNING DANN!"

accelerate launch \
    --gpu_ids=$GPU_IDS \
    --num_processes $GPUS \
    --mixed_precision $MIXED_PRECISION \
    --dynamo_backend "no" \
    $MULTI \
run_training.py \
    --config $CONFIG \
    --config_model $CONFIG_MODEL \
    --data_src $DATA_SRC \
    --data_tgt $DATA_TGT \
    --n_classes $N_CLASSES \
    --labels_file $LABELS \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulate_every $GRAD_ACC \
    --test_data_path $DATA_TEST >out_ours.txt


