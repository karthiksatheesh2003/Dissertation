# Leveraging AI for Environmental Acoustic Analysis

MSc Artificial Intelligence dissertation, University of Sheffield (2025)
Supervisor: Prof. Jon Barker

A convolutional neural network (CNN) that predicts urban acoustic loudness (A-weighted dBA) from mel-spectrograms, trained on recordings from the Silent Cities dataset.

## Overview

The COVID-19 lockdowns created an unusual chance to study city soundscapes. The Silent Cities project collected over 16 million 10-second recordings from 317 sites worldwide. This project asks: can a CNN estimate how loud a recording is from its spectrogram alone?

I used a subset of **15,000 recordings** and built a pipeline from raw audio through to a trained regression model.

## Results

| Model | R² | MAE (dBA) |
|---|---|---|
| Baseline CNN | below 0 | 7.73 |
| **Improved CNN (final)** | **0.734** | **4.81** |

Evaluated on a held-out validation set of 24,000 samples (80/20 split).

## Method

**Preprocessing**
- Mono conversion, resampling from 48 kHz to 16 kHz, and normalisation to 10 seconds
- A-weighted loudness (IEC 61672-1) computed in the frequency domain as the regression target

**Features**
- 64-band mel-spectrograms (1024-point FFT, hop length 256, 20-8000 Hz)
- Delta (first-order temporal derivative) features
- 4-channel input: paired adjacent patches, each with log-mel and delta channels

**2-second patching**
- Each 10-second spectrogram is cut into overlapping 2-second patches (50% overlap)
- 15,000 files became 135,000 patches and 120,000 paired training samples
- This cut memory use roughly fivefold per sample, allowing a batch size of 32 instead of 4-6

**Model**
- 3 convolutional blocks (4→32→64→128 channels; 5×5 then 3×3 kernels), each with batch normalisation, ReLU and max pooling
- Adaptive average pooling to 4×4, then a fully connected head (2048→256→64→1)
- Dropout of 0.5-0.7, about 8 million parameters

**Training**
- Huber loss (delta = 1.0) plus L1 regularisation (weight 0.01)
- AdamW, learning rate 5e-4, weight decay 1e-3, cosine annealing with warm restarts
- Batch size 32; peak validation performance at epoch 9
- Trained on CPU nodes of the University of Sheffield Stanage HPC cluster

## Tech stack

Python 3.8, PyTorch 1.12, librosa 0.9.2, scikit-learn 1.1

## Limitations

- Trained on a 15,000-file subset, not the full dataset
- CPU-only training limited model depth and batch size
- Predicts loudness only, not other acoustic indices
- 2-second patches may miss very short or transient sound events
- Validation used a random 80/20 split, not a split by site or time period, so generalisation to unseen locations is untested

## Future work

GPU training, attention mechanisms, transfer learning from pre-trained audio models, multi-task prediction of other acoustic indices, evaluation on the full dataset, and cross-dataset validation.

## Dataset

Silent Cities dataset: Challéat, S. et al. *A dataset of acoustic measurements from soundscapes collected worldwide during the COVID-19 pandemic.* Scientific Data 11, 928 (2024). The audio is not included in this repository.

## Repository structure

[Update to match your repo, for example: preprocessing scripts, model definition, training script, evaluation notebook, requirements.txt]

## Author

Karthik Satheesh
LinkedIn: https://www.linkedin.com/in/karthik-satheesh
