################################## ROOT PATHS ###########################################
ROOT="/mnt/hpc_data/shared_sim2real/"
ROOT_DATASETS=${ROOT}datasets/
ROOT_CKPT=${ROOT}ckpts/diffusion/

################################ CURRENT PATHS ##########################################
DATASET_PATH=${ROOT_DATASETS}databases/multi_rotations_and_forces/019_pitcher_base/no_bg
OUTPUT_ROOT=${ROOT_DATASETS}databases/multi_rotations_and_forces/019_pitcher_base/sim_mask3/
OUTPUT_BASE_NAME=converted

# uncomment for conditional diffusion
#CONDITION_FOLDER=${ROOT_DATASETS}masks/multi_rotations_and_forces/019_pitcher_base/sim_mask3/

CKPT_PATH=${ROOT_CKPT}sim2real_nobg_cond2/model-120.pt
CONFIG_PATH=${ROOT_CKPT}sim2real_nobg_cond2/train_diffusion.yaml

################################## HPARAMS ##############################################
GPUS=4
GPU_IDS="all"
BATCH_SIZE=40

NOISE_STEPS=( 100 )
SAMPLING_STEPS=( 50 )

#########################################################################################

if (( $GPUS > 1 )) ; then
    MULTI="--multi_gpu"
else
    MULTI=""
fi

for i in ${!NOISE_STEPS[@]}; do

    CURRENT_NOISE=${NOISE_STEPS[$i]}
    CURRENT_SAMPLING=${SAMPLING_STEPS[$i]}
    OUT_PATH=${OUTPUT_ROOT}${OUTPUT_BASE_NAME}_N${NOISE_STEPS}_S${CURRENT_SAMPLING}_R${REPETITIONS}/

    accelerate launch \
        --gpu_ids $GPU_IDS \
        --num_processes $GPUS \
        --mixed_precision "no" \
        --num_machines 1 \
        $MULTI \
    convert_dataset.py \
        --dataset_path $DATASET_PATH \
        --out_path $OUT_PATH \
        --ckpt_path $CKPT_PATH \
        --config_path $CONFIG_PATH \
        --noise_steps $CURRENT_NOISE \
        --sampling_steps $CURRENT_SAMPLING \
        --batch_size $BATCH_SIZE \
        #--condition_folder $CONDITION_FOLDER
done

