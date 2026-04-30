model_name=DD_STFF

python -u run.py \
  --grid_size 5 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Vegetables.csv \
  --model_id Vegetables_96_2 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len 2 \
  --e_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 512 \
  --d_core 256 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --grid_size 5 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Vegetables.csv \
  --model_id Vegetables_96_12 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1



python -u run.py \
  --grid_size 5 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Vegetables.csv \
  --model_id Vegetables_96_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --d_core 128 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --grid_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Vegetables.csv \
  --model_id Vegetables_96_48 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --d_core 128 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1



python -u run.py \
  --grid_size 5 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Vegetables.csv \
  --model_id Vegetables_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1


python -u run.py \
  --grid_size 4 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Vegetables.csv \
  --model_id Vegetables_96_12 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 512 \
  --d_core 256 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 10 \
  --lradj cosine \
  --des 'Exp' \
  --itr 1