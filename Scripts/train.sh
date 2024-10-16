python ../../run_self_training_ner.py \
  --data_dir data/CellLine \
  --train_file train_BIOUL.txt \
  --dev_file dev_BIOUL.txt \
  --test_file test_BIOUL.txt \
  --labels labels/CellLine.txt \
  --model_type bert \
  --model_name_or_path pubmedbert-uncased \
  --weight_decay 0 \
  --adam_epsilon 1e-8 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --warmup_steps 0 \
  --per_gpu_train_batch_size 160 \
  --per_gpu_eval_batch_size 160 \
  --logging_steps 100 \
  --save_steps 10000 \
  --do_train \
  --evaluate_during_training \
  --output_dir output \
  --max_seq_length 128 \
  --overwrite_output_dir \
  --self_training_begin_step 200 \
  --self_training_period 100 \
  --data_cahce_index pubmedbert-uncased \
  --visible_device 0 \
  --label_index 1 \
  --dataset self \
  --num_train_epochs 50 \
  --learning_rate 5e-5 \
  --seed 0 \
  --entity_ratio 0.01 \
  --entity_ratio_margin 0.05 \
  --update_scheme update_all \
  --entity_removal_method remove_annotations_randomly \
  --entity_removal_rate 0.9 \
  --self_training_loss_weight 10.0 \
  --prior_loss_weight 10.0 \
  --overall_eer_weight 0.5 \
