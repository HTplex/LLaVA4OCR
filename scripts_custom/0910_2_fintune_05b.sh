# vit_l_336/14 qwen2 0.5b, lrs3 336/28, 
# 8x h100, 

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "/home/agent_h/data/data_new/llms/Qwen2-0_5B" \
    --version plain \
    --data_path "/home/agent_h/data/data_new/datasets/facetalk_train/0906_lrs3_train/labels.json" \
    --image_folder "/home/agent_h/data/data_new/datasets/facetalk_train/0906_lrs3_train/images" \
    --vision_tower "/home/agent_h/data/data_new/llms/clip-vit-large-patch14-336" \
    --pretrain_mm_mlp_adapter /home/agent_h/data/data_new/llms-exp/0910_1_pretrain_05b/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/agent_h/data/data_new/llms-exp/0910_2_fintune_05b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 6400 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb