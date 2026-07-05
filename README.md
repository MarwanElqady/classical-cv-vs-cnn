# Ball Detection & Classification — Classical CV vs. Custom CNN

A six-class ball classifier built twice: once with a fully hand-crafted classical image processing pipeline, and once with a custom convolutional neural network designed from scratch (no pretrained models allowed). Course project for **ENMR602 — Signals & Image Processing**, German International University (GIU), Spring 2025/2026.


## The task

Given an RGB image containing a single ball — arbitrary background, lighting, and scale — classify it as one of six classes: **tennis ball, golf ball, football, basketball, baseball, or American football**. The dataset contains 1,696 images (1,407 train / 289 test) and was used exactly as provided, with the train/test split preserved.

## Results

| # | Model | Feature vector | Test accuracy |
|---|-------|----------------|---------------|
| 1 | Nearest-Class-Mean (baseline) | 3,584-dim | 51.21% |
| 2 | k-NN (k=1) | 3,584-dim | 72.66% |
| 3 | MLP (sklearn, PCA-256) | 3,584-dim | 78.55% |
| 4 | MLP (PyTorch, PCA-256) | 3,584-dim | 79.93% |
| 5 | RBF SVM (PCA-256) | 3,584-dim | 83.04% |
| 6 | Linear SVM (C=0.01) | 3,584-dim | 85.81% |
| 7 | Linear SVM + LBP | 3,610-dim | 87.54% |
| 8 | **Random Forest + LBP (300 trees)** | 3,610-dim | **89.97%** |
| 9 | **Custom CNN V4 (from scratch)** | raw pixels | **97.58%** |


## Approach 1 — Classical pipeline (Milestone 2)

No neural networks: every stage is designed by hand and justified from the data.

**Pipeline:** Read → Resize (256×192) → Gaussian blur → HSV conversion → Segment → Extract features → Classify

**Multi-strategy segmentation.** A per-class HSV diagnostic showed that only two of six classes have a usable colour signature, so a single thresholding method was never going to work. Four candidate paths compete per image — HSV `inRange`, Otsu (three variants), GrabCut with a central rectangle prior, and a central-crop fallback — and the winner is chosen by a mask quality score combining area plausibility, compactness, and a border-touch penalty. Over the full training set: Otsu wins 56.8%, HSV 30.8%, GrabCut 11.5%, fallback 0.9%, with only 0.9% of masks judged unusable.

**3,610-dimensional feature vector.** Masked HSV colour histograms (48), dual HOG descriptors on the mask bounding box and central crop (2×1,764), shape statistics with Hu moments (8), and masked Local Binary Patterns (26).

**Key ablation findings.** The 48-dim masked colour histogram alone reaches 88.93% — nearly matching the full pipeline. LBP adds a consistent +1.4 points and fixed the hardest confusions: golf ball F1 rose to 0.9655 (perfect precision) and football recall from 80% to 91% because LBP resolves dimples and stitching that HOG at 16×16-pixel cells cannot see.


## Approach 2 — Custom CNN (Milestone 3)

Course rules prohibited predefined architectures and pretrained weights — no ResNet, no transfer learning, no `tf.keras.applications`. The final model, **Custom CNN V4 SE HighRes**, was configured layer by layer:

- 256×256×3 input with training-only augmentation (flip, rotation, zoom, contrast)
- Conv stem + five stages of residual blocks with manually implemented squeeze-excitation channel attention (32 → 64 → 128 → 192 → 256 → 384 filters)
- Global average pooling head with Dense 512 → 256 → 6 softmax
- 6.93M parameters, AdamW with weight decay, label smoothing, early stopping — trained in 25.6 minutes

**Result: 97.58% test accuracy** (282/289 correct), macro F1 0.9757, with basketball and tennis ball at 100% recall. The +7.61-point jump over the classical pipeline came almost entirely from resolving the basketball ↔ American football confusion, which shares colour, silhouette, and coarse texture — exactly the case where learned features on raw pixels beat hand-crafted descriptors.


## Repository structure

```
notebooks/       Milestone 2 (classical pipeline) and Milestone 3 (CNN) notebooks
reports/         Milestone 1, 2, and 3 PDF reports
presentation/    Final evaluation slide deck
```

> **Note:** the dataset was provided by the course and is not redistributed here. The notebooks expect the original `dataset-balls/train` and `dataset-balls/test` folder structure.

## Team

Team A+ — Robotics and Automation Engineering, GIU:
Marwan Hesham Elqady · Moaaz Bassouny · Ahmed Mohamed Ramadan · Youssef Mokhtar 

Supervised by Dr. Moheb Mekhaiel, Eng. Moaz Fouda, Eng. Hisham Hafez, and Eng. Yasmine Attia.

## Tech stack

Python · OpenCV · scikit-learn · scikit-image · PyTorch · TensorFlow/Keras · NumPy · Matplotlib
