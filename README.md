# DSDIR
The source code of paper: "DSDIR: A Two-Stage Framework for Addressing Noisy Long-Tailed Problems in Malicious Traffic Detection"


# Step 1: data process

Preprocess the dataset: extract features and generate noise labels.

python3 data_process.py --dataset_name BoAu --corruption_type asym

# Step 2: Distribution-aware clean sample Selection (DS)

Based on the preprocessed dataset, use Masked Autoencoder Distribution Estimation (MADE) for clean sample selection. A clean subset C and a unlabeled subset U are obtained by DS.

python3 DS.py --dataset malicious_TLS-2023 --corruption_type asym --epochs 100 --corruption_ratio 0.0

# Step 3: Dynamic Instance-based Relabeling (DIR)

Use C and U as the basis, train a model and improve the quality of the dataset by using DIR.

python3 DIR.py --dataset_name malicious_TLS-2023 --corruption_ratio 0.8 --select_ratio 0.4 --beta 0.99 --epochs 100 
