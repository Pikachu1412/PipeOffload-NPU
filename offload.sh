#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b
export VSCODE_DEBUG=0
export CONNECT_RANKS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

pip install pulp sentencepiece

DATASET_DIR='/tmp/zb_sample_dataset'
DATASET="${DATASET_DIR}/dataset/c4_text_document"
TOKENIZER="${DATASET_DIR}/tokenizers/tokenizer.model"

if [ ! -e "$DATASET"".idx" ]; then
  if [ ! -e "zb_sample_dataset.tar.gz" ]; then
    wget https://huggingface.co/datasets/ufotalent/zero_bubble_sample_dataset/resolve/main/zb_sample_dataset.tar.gz
  fi
  tar -xvf zb_sample_dataset.tar.gz -C /tmp
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Running locally
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=5678
GPUS_PER_NODE=4
EXIT_INTERVAL=10
LOG_INTERVAL=10
WORLD_SIZE_IN_GPUS=$(( $WORLD_SIZE * $GPUS_PER_NODE ))

PIPELINE_SIZE=4
LAYERS=$(( $PIPELINE_SIZE * 2))
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$(( 8 ))
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=16384
ATTENTION_HEADS=32
GQA=8
TP_SIZE=1
SEQ_LENGTH=$((4096))

# profile_ranks="0"
# for ((i = 1; i < $WORLD_SIZE_IN_GPUS; i++)); do
#     profile_ranks="$profile_ranks $i"
# done


EVAL_INTERVAL=10000




# FFN_HIDDEN_SIZE=$(( $HIDDEN_SIZE * 4 ))


TRAIN_SAMPLES=$(( 146484375 * 1024 / $SEQ_LENGTH ))
LR_DECAY_SAMPLES=$(( 126953125 * 1024 / $SEQ_LENGTH ))
options=" \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PIPELINE_SIZE \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --ffn-hidden-size $FFN_HIDDEN_SIZE \
  --num-attention-heads $ATTENTION_HEADS \
  --group-query-attention \
  --num-query-groups $GQA \
  --exit-interval $EXIT_INTERVAL \
  --seq-length $SEQ_LENGTH \
  --max-position-embeddings $SEQ_LENGTH \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-samples $TRAIN_SAMPLES \
  --lr-decay-samples $LR_DECAY_SAMPLES \
  --lr-warmup-samples 183105 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-decay-style cosine \
  --log-interval ${LOG_INTERVAL} \
  --eval-iters 40 \
  --eval-interval $EVAL_INTERVAL \
  --data-path ${DATASET} \
  --tokenizer-type GPTSentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER} \
  --split 98,2,0 \
  --clip-grad 8.0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.006 \
  --no-barrier-with-level-1-timing \
  --transformer-impl local \
  --no-bias-dropout-fusion \
  --no-create-attention-mask-in-dataloader \
  --untie-embeddings-and-output-weights \
  --initial-loss-scale 65536 \
  --measure-activation-memory \
  --use-tp-pp-dp-mapping \
  --offload-continuous-buffers \
  --bf16 \
  --use-distributed-optimizer 
  "
OFFLOAD_ARGS=(
  --tensorboard-dir ./logs/test
  # --cpu-offload 
  --kaimm-offload-activation-ratio 0.5
  --bind-cpu
  --optimizer-selection fused_torch_adamw 
  --offload-overlap-sr
  # --sequence-parallel 

  --use-legacy-models
  --ckpt-format torch
  --profile
  --profile-ranks 0 1
  --use-pytorch-profiler
  --profile-step-start 3
  --profile-step-end 5
)

# 将OFFLOAD_ARGS拼接到options中
options="$options ${OFFLOAD_ARGS[@]}"

# if [ ! -z "$PROFILED" ]; then
#   options="$options --profile"
# fi




if [ ! -z "$ENABLE_EXACTLY_NUMERIC_MATCH" ]; then
  options="$options --enable-exactly-numeric-match \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0"
fi

# if [ ! -z "$INTERLEAVED_1F1B" ]; then
#   options="$options --num-layers-per-virtual-pipeline-stage 1"
#   if [ ! -z "$INTERLEAVE_GROUP" ]; then
#     options="$options --interleave-group-size $INTERLEAVE_GROUP --enable-zb-runtime"
#     if [ ! -z "$OFFLOAD" ]; then
#       if [ ! -z "$OFFLOAD_TIME" ]; then
#         options="$options --offload-time $OFFLOAD_TIME --offload-chunk-num $OFFLOAD_CHUNK_NUM"
#       else
#         options="$options --auto-offload-time --offload-chunk-num $OFFLOAD_CHUNK_NUM"
#       fi
#     fi
#   else
#     options="$options --no-pre-communication-optimization"
#   fi 
# else
#   options="$options --no-pre-communication-optimization"
# fi




run_cmd="torchrun --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE ${DIR}/pretrain_gpt.py $@ ${options} ${EXTRA_OPTIONS}"

if [ ! -z "$PROFILED" ]; then
  run_cmd="nsys profile -s none -t nvtx,cuda \
    --output $RANK.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $run_cmd"
fi

echo $run_cmd | tee log
# sleep 100000
eval $run_cmd 2>&1 | tee -a log

set +x
