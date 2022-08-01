WS_DIR=$HOME/workspace/discvae

python $WS_DIR/scripts/run_discvae.py \
    --model=discvae \
    --mode=train \
    --latent_size=128 \
    --dynamic_latent_size=32 \
    --hidden_size=512 \
    --rnn_size=128 \
    --encoded_latent_size=32 \
    --batch_size=16 \
    --logdir="$WS_DIR/checkpoints" \
    --random_seed=7 \
    --learning_rate=0.0003 \
    --num_epochs=70 \
    --save_every=1 \
    --gpu_num="0" \
    --split=valid \
    --num_samples=4 \
    --dataset=moving_mnist \
    --dataset_path="$WS_DIR/data" \
    --filename="moving_mnist" \
    --seq_length=20