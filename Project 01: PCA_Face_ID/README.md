# PCA Face Identification & Model Robustness

## 📌 Project Overview
This repository contains **Team 5's** machine learning project focused on Facial Recognition using Principal Component Analysis (PCA) and evaluating model robustness (مقاومت مدل) against noise and partial occlusion[cite: 1]. The model is trained on the Olivetti dataset, consisting of 64x64 grayscale facial images (flattened to 4096 pixels with values normalized between 0.0 and 1.0)[cite: 1].

## ⚙️ The Core Pipeline
Our baseline facial recognition system compresses high-dimensional image data into a smaller subspace and identifies individuals using geometric proximity:
* **Data Preparation:** Calculates a `mean_face` from the training data and centers the dataset by subtracting this average[cite: 1].
* **Eigendecomposition:** Extracts principal components (Eigenfaces) using the covariance matrix and `np.linalg.eigh`, sorting the eigenvalues in descending order of importance[cite: 1].
* **SVD Validation:** Proves the mathematical integrity of the subspace by comparing the Covariance/Eigh path against a Singular Value Decomposition ($A = U \Sigma V^T$) shortcut, ensuring reconstruction errors between the two are near zero[cite: 1].
* **Classification:** Uses a 1-Nearest Neighbor (1-NN) algorithm to predict identities by finding the shortest Euclidean distance between the compressed weights of the test and training images[cite: 1].

## 🧮 Core Mathematics
The project relies on two primary mathematical operations for dimensionality reduction:
* **Projection (Compression):** A mean-centered test image ($X_{centered}$) is compressed down to $k$ scalar weights by calculating its dot product with the top $k$ eigenvectors ($U_k$)[cite: 1]. 
  $$W = X_{centered} \cdot U_k$$
* **Reconstruction:** To rebuild the image and calculate Mean Squared Error (MSE), the $k$ weights ($W$) are multiplied by the transpose of the eigenvectors ($U_k^T$), and the mean face is added back[cite: 1]. 
  $$X_{reconstructed} = (W \cdot U_k^T) + mean\_face$$

## 🛡️ Robustness Testing (Team 5 Special Mission)
To evaluate where the PCA model breaks down, we applied targeted disruptions **exclusively to the test data**, ensuring the eigenfaces were learned purely from clean images[cite: 1].

### 1. Environmental Noise
We tested model resilience against three real-world noise distributions across varying intensities[cite: 1]:
* **Gaussian Noise:** Simulates sensor static by adding values drawn from a normal distribution[cite: 1]. 
  $$X_{noisy} = X + \mathcal{N}(\mu, \sigma)$$
* **Salt & Pepper Noise:** Simulates dead camera pixels by forcefully overwriting a random percentage of pixels to pure white ($1.0$) or pure black ($0.0$)[cite: 1].
* **Poisson (Shot) Noise:** Simulates the quantum nature of light in dark environments where the variance is tied directly to the pixel's original brightness ($peak$)[cite: 1]. 
  $$X_{noisy} = \frac{\mathcal{P}(X \cdot peak)}{peak}$$

### 2. Occlusion (Masking)
We simulated physical obstructions by overwriting targeted matrices of pixels with pure black ($0.0$)[cite: 1]:
* **Fixed Real-World Masks:** Tested a "Sunglasses" mask (blocking rows 15-35), a "Medical Mask" (blocking rows 35-64), and a "Half-Face" shadow (blocking the first 32 columns)[cite: 1].
* **Random Sliding Blocks:** Dropped black squares ranging from tiny 5x5 pixels up to massive 50x50 pixels on random coordinates to map out the exact geometric failure points in high-resolution heatmaps[cite: 1].

## 📊 Key Findings & Insights
* **Dimensionality Limits:** The model's accuracy naturally plateaus around $k=50$, demonstrating that we can discard thousands of components and still retain core facial geometry[cite: 1]. Furthermore, the maximum number of meaningful components is strictly mathematically bounded by the 320 training samples[cite: 1].
* **Global vs. Local Failure:** Standard PCA is fundamentally a "global" algorithm and fails catastrophically against occlusion because it processes the black mask pixels as actual facial structure rather than ignoring them[cite: 1].
* **Feature Importance:** The model relies significantly heavier on the geometry of the eyes and eyebrows than the lower face; accuracy dropped more severely with the Sunglasses mask (~8%) than with the Medical Mask (~12%)[cite: 1].
* **Noise Sensitivity:** Salt & Pepper noise destroys structural data and plummets accuracy incredibly fast, whereas the model remains highly resilient to Poisson noise[cite: 1].
