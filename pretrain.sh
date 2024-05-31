#!/bin/bash

# llava original with qwen1.5-0.5b
deepspeed llava/train/train_mem.py \
    --data_path /data/agent_h/vsr2/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /data/agent_h/vsr2/datasets/LLaVA-Pretrain/images \
    --output_dir /data/agent_h/vsr2/checkpoints/llava-1.5-v0.1-0.8b-pretrain \
    --vision_tower /data/agent_h/vsr2/llms/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --model_name_or_path /data/agent_h/vsr2/llms/Qwen1.5-0.5B \
    --deepspeed ./scripts/zero1.json \
    --version plain \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
