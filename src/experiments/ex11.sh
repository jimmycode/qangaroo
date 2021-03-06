export CUDA_VISIBLE_DEVICES=5
python run.py \
  --model=multi_hhn \
  --data_path=data/wikihop/train.json.proc \
  --input_vocab=data/wikihop/train.json.encall.vocab \
  --type_vocab=data/wikihop/train.json.type.vocab \
  --input_vsize 200000 \
  --type_vsize 280 \
  --ckpt_root=checkpoints/multi_hhn/ex11 \
  --summary_dir=log/multi_hhn//ex11 \
  --mode=train \
  --batch_size 16 \
  --lr 0.1 \
  --min_lr 0.001 \
  --max_grad_norm 4.0 \
  --decay_step 10000 \
  --decay_rate 0.6 \
  --max_run_steps 100000 \
  --dropout 0.1 \
  --valid_path=data/wikihop/dev/dev.json.proc \
  --valid_freq 500 \
  --num_valid_batch 30 \
  --checkpoint_secs 1200 \
  --display_freq 100 \
  --use_bucketing False \
  --truncate_input True \
  --num_gpus 1 \
  --emb_dim 64 \
  --type_emb_dim 64 \
  --max_num_doc 64 \
  --max_doc_len 200 \
  --max_entity_len 10 \
  --max_num_cands 80 \
  --word_conv_filter 128 \
  --word_conv_width 3 \
  --hop_net_rnn_layers 1 \
  --hop_net_rnn_num_hid 64 \
  --num_hops 1 \
  --hop_mod_reuse True \

