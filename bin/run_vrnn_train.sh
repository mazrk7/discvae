WS_DIR=$HOME/workspace/discvae

for i in 0 1 2 3 4 5 6 7 8 9
do
    python $WS_DIR/scripts/run_discvae.py \
        --model=vrnn \
        --mode=train \
        --latent_size=256 \
        --hidden_size=512 \
        --rnn_size=64 \
        --encoded_latent_size=32 \
        --batch_size=16 \
        --logdir="$WS_DIR/checkpoints" \
        --random_seed=$i \
        --learning_rate=0.0003 \
        --num_epochs=40 \
        --save_every=1 \
        --gpu_num="0" \
        --split=test \
        --num_samples=4 \
        --dataset=sprites \
        --dataset_path="$WS_DIR/data" \
        --seq_length=8 \
        --num_channels=3
done