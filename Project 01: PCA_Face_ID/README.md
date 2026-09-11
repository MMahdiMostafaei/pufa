# PCA Face Identification & Model Robustness

## 📌 Project Overview
This repository contains **Team 5's** machine learning project focused on Facial Recognition using Principal Component Analysis (PCA) and evaluating model robustness against noise and partial occlusion. The model is trained on the Olivetti dataset, consisting of 64x64 grayscale facial images (flattened to 4096 pixels with values normalized between 0.0 and 1.0).

## ⚙️ The Core Pipeline
Our baseline facial recognition system compresses high-dimensional image data into a smaller subspace and identifies individuals using geometric proximity:
* **Data Preparation:** Calculates a `mean_face` from the training data and centers the dataset by subtracting this average.
* **Eigendecomposition:** Extracts principal components (Eigenfaces) using the covariance matrix and `np.linalg.eigh`, sorting the eigenvalues in descending order of importance.
* **SVD Validation:** Proves the mathematical integrity of the subspace by comparing the Covariance/Eigh path against a Singular Value Decomposition ($A = U \Sigma V^T$) shortcut, ensuring reconstruction errors between the two are near zero.
* **Classification:** Uses a 1-Nearest Neighbor (1-NN) algorithm to predict identities by finding the shortest Euclidean distance between the compressed weights of the test and training images.

## 🧮 Core Mathematics
The project relies on two primary mathematical operations for dimensionality reduction:
* **Projection (Compression):** A mean-centered test image ($X_{centered}$) is compressed down to $k$ scalar weights by calculating its dot product with the top $k$ eigenvectors ($U_k$). 
  $$W = X_{centered} \cdot U_k$$
* **Reconstruction:** To rebuild the image and calculate Mean Squared Error (MSE), the $k$ weights ($W$) are multiplied by the transpose of the eigenvectors ($U_k^T$), and the mean face is added back. 
  $$X_{reconstructed} = (W \cdot U_k^T) + mean\_face$$

## 🛡️ Robustness Testing (Team 5 Special Mission)
To evaluate where the PCA model breaks down, we applied targeted disruptions **exclusively to the test data**, ensuring the eigenfaces were learned purely from clean images.

### 1. Environmental Noise
We tested model resilience against three real-world noise distributions across varying intensities:
* **Gaussian Noise:** Simulates sensor static by adding values drawn from a normal distribution. 
  $$X_{noisy} = X + \mathcal{N}(\mu, \sigma)$$
* **Salt & Pepper Noise:** Simulates dead camera pixels by forcefully overwriting a random percentage of pixels to pure white ($1.0$) or pure black ($0.0$).
* **Poisson (Shot) Noise:** Simulates the quantum nature of light in dark environments where the variance is tied directly to the pixel's original brightness ($peak$). 
  $$X_{noisy} = \frac{\mathcal{P}(X \cdot peak)}{peak}$$

### 2. Occlusion (Masking)
We simulated physical obstructions by overwriting targeted matrices of pixels with pure black ($0.0$):
* **Fixed Real-World Masks:** Tested a "Sunglasses" mask (blocking rows 15-35), a "Medical Mask" (blocking rows 35-64), and a "Half-Face" shadow (blocking the first 32 columns).
* **Random Sliding Blocks:** Dropped black squares ranging from tiny 5x5 pixels up to massive 50x50 pixels on random coordinates to map out the exact geometric failure points in high-resolution heatmaps.

## 📊 Key Findings & Insights
* **Dimensionality Limits:** The model's accuracy naturally plateaus around $k=50$, demonstrating that we can discard thousands of components and still retain core facial geometry. Furthermore, the maximum number of meaningful components is strictly mathematically bounded by the 320 training samples.
* **Global vs. Local Failure:** Standard PCA is fundamentally a "global" algorithm and fails catastrophically against occlusion because it processes the black mask pixels as actual facial structure rather than ignoring them.
* **Feature Importance:** The model relies significantly heavier on the geometry of the eyes and eyebrows than the lower face; accuracy dropped more severely with the Sunglasses mask (~8%) than with the Medical Mask (~12%).
* **Noise Sensitivity:** Salt & Pepper noise and Poisson noise destroy structural data and plummet accuracy incredibly fast, whereas the model remains more accurate subjected to Gaussian noise.
