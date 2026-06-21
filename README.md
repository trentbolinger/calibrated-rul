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
- Note: this Kaggle mirror has a minor discrepancy from the original NASA
  paper for FD004 -- 249 train engines / 248 test engines here, versus the
  paper's reported 248 train / 249 test. Verified this isn't a mislabeling
  issue: train trajectories average 246 cycles (consistent with
  run-to-failure data) vs. test's 166 cycles (consistent with truncated
  data), so the file labels are correct -- just a one-engine difference in
  this specific reupload.

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