WS_DIR=$HOME/workspace/discvae

python $WS_DIR/scripts/run_discvae.py \
    --model=gmvae \
    --mode=eval \
    --latent_size=128 \
    --hidden_size=512 \
    --batch_size=8 \
    --logdir="$WS_DIR/checkpoints" \
    --random_seed=7 \
    --gpu_num="0" \
    --split=test \
    --num_samples=4 \
    --dataset=moving_mnist \
    --dataset_path="$WS_DIR/data" \
    --filename="moving_mnist" \
    --seq_length=20