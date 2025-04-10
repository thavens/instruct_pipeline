# Loop through each checkpoint value and run the Python script
# for p in 0.3 0.5 0.7 0.9 1; do
#     for ckpt in {50..450..50}; do
#         echo "Running test_nq_sft.py with checkpoint: $ckpt and prob: $p"
#         python test_nq_sft.py --p $p --ckpt $ckpt
#     done
# done

p=0.5
for ckpt in {50..450..50}; do
    echo "Running test_nq_sft.py with checkpoint: $ckpt and prob: $p"
    python test_nq_sft.py --p $p --ckpt $ckpt
done