# DATASET TEST
DATA_TEST="/mnt/hpc_data/shared_sim2real/datasets/databases/final_testset/"

# CONFIG
CONFIG="config/dino_v2_test_config.yaml"
CKPT="results/YOUR_CKPT.ckpt"


#########################################################################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
# training info
GPUS=4

if (( $GPUS > 1 )) ; then
    MULTI="--multi_gpu"
else
    MULTI=""
fi

echo "STARTING TEST OF MODEL!"

accelerate launch \
    --gpu_ids="all" \
    --num_processes $GPUS \
    --mixed_precision "no" \
    --dynamo_backend "no" \
    $MULTI \
run_eval.py \
    --config $CONFIG \
    --ckpt $CKPT \
    --data_path $DATA_TEST \
    --log out.csv

