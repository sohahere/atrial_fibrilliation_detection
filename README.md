# Deep Learning Pipeline for ECG-Based Classification of AF & PAF

A lightweight deep-learning pipeline for classifying **Atrial Fibrillation (AF), Paroxysmal Atrial Fibrillation (PAF), and Normal/Non-AF (NAF) rhythms** from short 10-second ECG recordings.

The system converts raw ECG signals into **CQT time-frequency spectrograms** and uses a **MobileNetV2 + Cross-Scale Attention (CSA) + SimAM** architecture to learn discriminative features for rhythm classification.

---

## Overview

Atrial Fibrillation is an irregular cardiac rhythm associated with serious complications such as stroke and heart failure. Detecting **Paroxysmal Atrial Fibrillation (PAF)** is particularly challenging because episodes can be brief and intermittent, making them easy to miss in short ECG recordings.

This project addresses the problem by building an end-to-end deep-learning pipeline that:

- Processes raw ECG recordings
- Applies signal preprocessing and noise filtering
- Segments ECG recordings into **10-second windows**
- Converts ECG signals into **224×224 RGB CQT spectrograms**
- Extracts features using lightweight **MobileNetV2**
- Enhances features using **Cross-Scale Attention (CSA)**
- Uses **SimAM** for parameter-free spatial attention
- Classifies the input into **AF, PAF, or NAF**

The overall objective is to improve the detection of subtle PAF patterns while keeping the architecture computationally lightweight.

---

## Pipeline

```text
                    ECG Recording
                         │
                         ▼
                Signal Preprocessing
                         │
                ┌────────┴────────┐
                │                 │
          Bandpass Filter      10-sec
           0.5–45 Hz           Windows
                │                 │
                └────────┬────────┘
                         ▼
                  CQT Transformation
                         │
                         ▼
              224 × 224 RGB Spectrogram
                         │
                         ▼
                    MobileNetV2
                  Feature Extraction
                         │
                         ▼
                Feature Refinement
                         │
                  5×5 Separable Conv
                         │
                         ▼
              Cross-Scale Attention
                    ┌────┴────┐
                    │         │
               Multi-scale  SimAM
                features    Attention
                    │         │
                    └────┬────┘
                         ▼
                    Feature Fusion
                         │
                         ▼
                Global Average Pooling
                         │
                         ▼
                    Softmax Classifier
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
             AF         PAF        NAF
```

---

## Dataset

The project uses data from the **CPSC 2021 Paroxysmal AF Detection Challenge**.

The dataset contains dynamic ECG recordings collected using **12-lead Holter devices and 3-lead wearable monitors**. The available recordings are variable-length ECG segments from leads I and II, sampled at **200 Hz**.

Two training sets provided by the challenge were merged to increase the diversity and quantity of training data.

### Target Classes

| Class | Description |
|---|---|
| **AF** | Atrial Fibrillation |
| **PAF** | Paroxysmal Atrial Fibrillation |
| **NAF** | Normal / Non-AF rhythm |

PAF is the most challenging class because its short and intermittent episodes can have subtle differences from both AF and normal rhythms.

---

## Preprocessing

The raw ECG signals are processed before being passed to the deep-learning model.

### 1. Bandpass Filtering

A **2nd-order Butterworth bandpass filter** is applied with:

```text
Low cutoff  = 0.5 Hz
High cutoff = 45 Hz
```

This helps reduce baseline wander and high-frequency noise.

A notch filter was not used because no major 50/100 Hz interference was observed in the data.

### 2. ECG Segmentation

The filtered ECG recordings are divided into:

```text
10-second ECG windows
```

Short fixed-length segments make the input consistent for deep-learning training and inference.

### 3. CQT Transformation

Each 10-second ECG segment is converted from a 1-D signal into a **2-D time-frequency representation** using Constant-Q Transform (CQT).

```text
Raw ECG
   │
   ▼
CQT
   │
   ▼
224 × 224 RGB Spectrogram
```

CQT makes temporal and frequency-related patterns easier for convolutional networks to learn, particularly when distinguishing subtle rhythm irregularities.

---

# Model Architecture

The proposed model combines three major components:

```text
CQT Image
    │
    ▼
MobileNetV2
    │
    ▼
Feature Refinement
    │
    ▼
Cross-Scale Attention ───┐
                         ├──► Feature Fusion
SimAM Attention ─────────┘
                         │
                         ▼
                 Global Average Pooling
                         │
                         ▼
                   Softmax Classifier
                         │
                         ▼
                  AF / PAF / NAF
```

---

## 1. MobileNetV2

**MobileNetV2** is used as the lightweight backbone for feature extraction.

Input:

```text
224 × 224 RGB CQT image
```

The backbone extracts meaningful spatial and time-frequency features while keeping computational requirements relatively low.

After feature extraction, the architecture applies:

- **2×2 Average Pooling**
- **5×5 Separable Convolution**
- **512 filters**

These operations prepare the feature representation for the attention modules.

---

## 2. Cross-Scale Attention (CSA)

PAF patterns can appear at different spatial scales within the CQT representation.

The Cross-Scale Attention module divides the feature maps into channel groups and processes them independently.

Each group is passed through a **Sequential Attention (SA)** block containing SMDC layers with increasing kernel sizes.

Three types of feature processing are used:

```text
Generic       → Multi-scale patterns
Row-wise      → Horizontal patterns
Column-wise   → Vertical patterns
```

The outputs from these attention heads are concatenated to form a richer multi-scale feature representation.

---

## 3. SimAM Attention

The architecture also incorporates **SimAM (Simple Attention Module)**.

Unlike many conventional attention mechanisms, SimAM is parameter-free and uses neuron-energy-based attention to identify important spatial regions.

This allows the model to enhance informative features without introducing a large number of additional trainable parameters.

---

## 4. Feature Fusion

The outputs from CSA and SimAM are combined:

```text
              CSA Features
                   │
                   ├──────────┐
                   │          │
                   ▼          ▼
                         Feature Fusion
                              │
SimAM Features ───────────────┘
                              │
                              ▼
                     Global Average Pooling
                              │
                              ▼
                         Classifier
```

A **Global Average Pooling (GAP)** layer converts the final 2-D feature maps into a 1-D representation before classification.

---

# Classification

The final classifier uses a softmax layer to predict one of three rhythm classes:

```text
                    ┌─────────────┐
                    │   Softmax   │
                    └──────┬──────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
            AF            PAF           NAF
```

The model therefore performs **3-class ECG rhythm classification** rather than only binary AF detection.

---

# Results

The model demonstrated strong classification performance across AF, PAF, and NAF.

The training and validation accuracy curves converged with only a small gap between them, indicating stable learning and good generalization without obvious overfitting.

### PAF Classification

PAF was the most challenging class, yet the confusion matrix showed relatively few PAF misclassifications.

Out of the PAF samples:

- **16** were misclassified as AF
- **51** were misclassified as NAF
- **67 total PAF classification errors**

This represents an improvement over earlier versions of the model, which had greater difficulty distinguishing short and transient PAF episodes.

---

# Feature Space Analysis

A t-SNE visualization was used to examine the learned feature representations.

The resulting feature space showed distinct clusters for the three rhythm classes:

```text
             AF
          ● ● ● ●
       ● ● ● ● ●

                    PAF
                  ● ● ●
                ● ● ● ●

                         NAF
                      ● ● ● ●
                    ● ● ● ●
```

The PAF samples formed a separate cluster between AF and NAF, suggesting that the network learned discriminative features capable of separating the subtle PAF patterns from the other rhythm classes.

---

# Why This Architecture?

The main design idea is to combine **efficient feature extraction with complementary attention mechanisms**.

| Component | Purpose |
|---|---|
| **CQT** | Converts ECG into a time-frequency representation |
| **MobileNetV2** | Lightweight feature extraction |
| **CSA** | Captures patterns at multiple spatial scales |
| **SimAM** | Highlights informative regions without additional parameters |
| **GAP** | Reduces feature maps to a compact representation |
| **Softmax** | Performs 3-class rhythm classification |

The combination is designed to capture subtle, short-duration PAF characteristics while maintaining a relatively lightweight architecture.

---

# Key Features

- ECG preprocessing and filtering
- 10-second ECG segmentation
- CQT-based time-frequency representation
- 224×224 RGB input generation
- Lightweight MobileNetV2 backbone
- Cross-Scale Attention
- SimAM parameter-free attention
- Multi-scale feature learning
- AF / PAF / NAF classification
- Confusion matrix analysis
- Training/validation performance analysis
- t-SNE feature-space visualization

---

# Future Scope

The current pipeline can be extended in several directions:

- Evaluation on additional ECG datasets
- Multi-lead ECG analysis
- Real-time deployment on wearable devices
- Patient-level AF burden estimation
- Further optimization for edge/mobile deployment

These extensions could help move the system from an offline classification pipeline toward practical continuous cardiac monitoring

# End-to-End Workflow

```text
ECG Recordings
      │
      ▼
Bandpass Filtering
0.5–45 Hz
      │
      ▼
10-second Segmentation
      │
      ▼
CQT Transformation
      │
      ▼
224×224 RGB Images
      │
      ▼
MobileNetV2
      │
      ▼
Feature Refinement
      │
      ├───────────────┐
      ▼               ▼
     CSA            SimAM
      │               │
      └───────┬───────┘
              ▼
        Feature Fusion
              │
              ▼
      Global Average Pooling
              │
              ▼
          Softmax
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
     AF      PAF     NAF
```

---

# Project Information

**Project:** A Deep Learning Pipeline for ECG-Based Classification of Atrial Fibrillation

**Organization:** Biomedical Signal & Image Processing (BSIP) Lab, NIT Rourkela

**Author:** Soha Ghodeswar

**Guide:** Dr. Bala C Neelapu

The project was developed as a summer internship project focused on biomedical signal processing and deep learning using real physiological ECG data.
