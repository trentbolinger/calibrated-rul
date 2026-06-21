# calibrated-rul

## Overview
Predicting jet engine RUL with calibrated uncertainty, and testing whether that uncertainty holds up under distribution shift.

## Problem Statement
- What RUL prediction is and why it matters (predictive maintenance)
- Why uncertainty matters, not just point predictions
- The specific question: does conformal prediction's coverage guarantee
  survive when the model sees operating conditions it wasn't trained on?

## Dataset
- NASA C-MAPSS turbofan engine degradation simulation
- FD001 (single condition) for training/calibration
- FD002 / FD004 (six conditions) for the shift test
- Link: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## Method
- LSTM regression model for RUL prediction
- Conformal prediction for calibrated intervals
- Evaluation: RMSE, PICP (coverage), sharpness (interval width)

## Project Structure
[TBD]

## Setup / Usage
[TBD — how to run train.py / evaluate.py]

## Results
[TBD]

## Key Finding
[TBD]