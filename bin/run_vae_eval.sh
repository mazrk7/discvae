WS_DIR=$HOME/workspace/discvae

for i in 0 1 2 3 4 5 6 7 8 9
do
    python $WS_DIR/scripts/run_discvae.py \
        --model=vae \
        --mode=eval \
        --latent_size=128 \
        --hidden_size=512 \
        --batch_size=8 \
        --logdir="$WS_DIR/checkpoints" \
        --random_seed=$i \
        --gpu_num="0" \
        --split=test \
        --num_samples=4 \
        --dataset=moving_mnist \
        --dataset_path="$WS_DIR/data" \
        --filename="moving_mnist" \
        --seq_length=20
done
