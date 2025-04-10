p=0.5
for ckpt in {100..500..100}; do
    echo "Running test_nq_grpo.py with checkpoint: $ckpt and prob: $p"
    python test_nq_grpo.py --p $p --ckpt $ckpt
done


echo "Running test_nq_grpo.py with checkpoint: 570 and prob: $p"
python test_nq_grpo.py --p $p --ckpt 570