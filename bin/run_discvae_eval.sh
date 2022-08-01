WS_DIR=$HOME/workspace/discvae

python $WS_DIR/scripts/run_discvae.py \
    --model=discvae \
    --mode=eval \
    --latent_size=128 \
    --dynamic_latent_size=32 \
    --hidden_size=512 \
    --rnn_size=128 \
    --encoded_latent_size=32 \
    --batch_size=16 \
    --logdir="$WS_DIR/checkpoints" \
    --random_seed=7 \
    --gpu_num="0" \
    --split=test \
    --num_samples=4 \
    --prefix_length=10 \
    --sample_length=10 \
    --dataset=moving_mnist \
    --dataset_path="$WS_DIR/data" \
    --filename="moving_mnist" \
    --seq_length=20