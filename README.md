# DSDIR
The source code of paper: "DSDIR: A Two-Stage Framework for Addressing Noisy Long-Tailed Problems in Malicious Traffic Detection"


# Dependencies
pytorch >= 2.0.0  

cuda >= 11.8

The detailed environment set can be found in the requirements.txt

# Dataset

**malicious_TLS-2023**
https://github.com/gcx-Yuan/BoAu
**CIC-IDS-2017**
https://www.unb.ca/cic/datasets/ids-2017.html

# Training

First, preprocess the dataset: extract features and generate noisy labels.

'python3 main/data_process.py --dataset_name malicious_TLS-2023 --noise_type asym'

Second, Distribution-aware clean sample Selection and Dynamic Instance-based Relabeling (DSDIR)

'python3 main/DSDIR.py --dataset_name malicious_TLS-2023 --noise_type asym --noise_ratio 0.2 --select_ratio 0.4 --beta -1 --epochs 100 --warm_up 30'

dataset_name: the name of the dataset
noise_type: the type of noise scenario: asym/sym (asymmetric/symmetric)
noise_ratio: the ratio of noise
beta: the parameter of loss. -1 means the cross entropy; 0.9,0.99,... 0.999999 means the parameter of the class-balanced loss("Class-Balanced Loss Based on Effective Number of Samples").

And this implementation of MADE is copied from: https://github.com/e-hulten/made.
