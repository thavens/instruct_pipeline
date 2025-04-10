# Loop through each checkpoint value and run the Python script
for p in 0.3 0.5 0.7 0.9 1; do
    echo "Running train_nq_sft.py with $p probability"
    python train_nq_sft.py --p $p
done