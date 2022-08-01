WS_DIR=$HOME/workspace/discvae

python $WS_DIR/scripts/run_discvae.py \
    --model=vae \
    --mode=train \
    --latent_size=128 \
    --hidden_size=512 \
    --batch_size=8 \
    --logdir="$WS_DIR/checkpoints" \
    --random_seed=7 \
    --learning_rate=0.0003 \
    --num_epochs=55 \
    --save_every=1 \
    --gpu_num="0" \
    --split=valid \
    --num_samples=4 \
    --dataset=moving_mnist \
    --dataset_path="$WS_DIR/data" \
    --filename="moving_mnist" \
    --seq_length=20