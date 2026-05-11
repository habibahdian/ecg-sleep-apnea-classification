# ecg-sleep-apnea-classification
ECG-based sleep apnea classification using Multi-Scale SE-ResNet with 6 HRV features (RR intervals, R-peak amplitude, QRS amp std, Mean HR, Triangular Index, pNN50). Trained on NCKUHSC PSG dataset with StratifiedGroupKFold cross-validation. 


# ECG Sleep Apnea Classification — Inference Guide

## Files
- inference.py              → full inference pipeline
- model_6feat_fold2.pth     → trained model weights

## Model Info
- Task        : Binary classification (Normal / Apnea-Hypopnea)
- Model       : Multi-Scale SE-ResNet (6 input channels)
- Parameters  : ~7.5M (requires quantization for STM32 NPU)
- Performance : AUC=0.8531 | Acc=0.7782 | F1=0.7559 | Sens=0.7133 | Spec=0.8384 | ACC=0.7782

## Input Requirements
- Signal      : ECG
- Duration    : 60 seconds per segment
- Sampling    : 200 Hz (12000 samples)
- Format      : numpy array, shape (12000,)
- Note        : No preprocessing needed — bandpass and z-score 
                are applied automatically inside predict()

## Dependencies
pip install numpy torch scipy biosppy neurokit2

## Usage
from inference import load_model, predict
import numpy as np

model = load_model()
ecg_segment = np.load('your_ecg_segment.npy')  # shape: (12000,)
label, prob = predict(ecg_segment, model)
print(f"Prediction: {label} | Probability: {prob:.4f}")

## Output
- label      : "Normal" or "AH" (Apnea-Hypopnea)
- prob       : probability score (0.0 – 1.0)
- threshold  : 0.4472 (Youden Index optimal threshold)

## Preprocessing Pipeline (inside predict())
1. Bandpass filter  : 8–50 Hz (Butterworth)
2. Z-score          : per segment normalization
3. Feature extract  : RR intervals, R-peak amplitude, 
                      QRS amp std, Mean HR, 
                      Triangular Index, pNN50
4. Interpolation    : cubic spline → 240 points @ 4Hz
