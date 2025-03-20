export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="OpenVLA-7B-GRPO-REC-lora"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path openvla/openvla-7b \
    --dataset_name data_config/rec.yaml \
    --image_root <your_image_root> \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --run_name $RUN_NAME \
    --save_steps 5000 \
    --save_only_model false \
    --learning_rate 5e-4 \
    --use_peft true \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true


