#!/bin/bash

# Runs the "175B" parameter model
# export PYTHONPATH="/data0/ssy/MindSpeed/:$PYTHONPATH"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=0
# export ASCEND_LAUNCH_BLOCKING=1
# export ADAPTIVE_RECOMPUTING=1
# export ASCEND_RT_VISIBLE_DEVICES=6
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=5008
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=./ckpts
TENSORBOARD_LOGS_PATH=./logs
VOCAB_FILE=/data0/ssy/data/gpt2-vocab.json 
MERGE_FILE=/data0/ssy/data/gpt2-merges.txt 
DATA_PATH=/data0/ssy/data/my-gpt2_text_document 

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 16
    --hidden-size 4096
    --num-attention-heads 32 
    --seq-length 2048
    --max-position-embeddings 2048 
)

TRAINING_ARGS=(
    --bf16
    --micro-batch-size 1 
    --global-batch-size 8 
    #--rampup-batch-size 16 16 5859375 
    --train-iters 20
    # --weight-decay 0.1 
    # --adam-beta1 0.9 
    # --adam-beta2 0.95 
    # --init-method-std 0.006 
    # --clip-grad 1.0 
    
    --lr 0.1
    # --lr-decay-style cosine 
    # --min-lr 6.0e-6
    # --lr-warmup-fraction .001 
    # --lr-decay-iters 430000 
    --use-legacy-models
    --ckpt-format torch 
    # --optimizer-selection fused_torch_adamw
    # --distribute-saved-activations
    # --adaptive-memory-optimization
    # --prefetch-layer

    # --automated-pipeline
    # --adaptive-recompute-device-swap
    # --adaptive-recompute-device-size  61440 
    # --recompute-activation-function # 激活函数重计算
    # --recompute-activation-function-num-layers 10  # 激活函数重计算层数 
    # --group-query-attention
    # --num-query-groups 4 
    # --recompute-activations       # megatron选择重计算
    # --recompute-granularity full    # 开启完全重计算
    # --recompute-method block     # 配置具体重计算方式
    # --recompute-num-layers 5
    # --memory-fragmentation
)

MODEL_PARALLEL_ARGS=(
	# --tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 4

)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    # --log-interval 10
    --save-interval 10
    # --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    # --eval-iters 10
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

PROFILING_ARG=(
    --profile	#打开profile开关
    --profile-step-start 15	#配置开始采集步，未配置时默认为10, 配置举例: --profile-step-start 30
    --profile-step-end 19
    --profile-level level2
    --profile-with-cpu
    --profile-with-memory
    --profile-save-path=./profile_dir
)

torchrun ${DISTRIBUTED_ARGS[@]} /data0/ssy/Megatron-LM-0.9.0/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${PROFILING_ARG[@]} 
    # ${EVAL_AND_LOGGING_ARGS[@]}
