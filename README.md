# Principal Component Analysis from Scratch in Python

This repository provides a step-by-step implementation of **Principal Component Analysis (PCA)** in Python.
In general, PCA is a method for reducing the number of features in a dataset while keeping the most important information. Some datasets contain a large number of features, and many of them may be redundant, noisy, or less useful than others. Dimensionality reduction helps us remove this unnecessary information.
In mathematical terms, the goal of PCA is to find an optimal $m$-dimensional affine subspace of a high-dimensional space $\mathbb{R}^n$ ($m < n$) that best approximates the data, i.e. the one that preserves the maximum variance (equivalently, minimizes the reconstruction error).

---
## Mathematical Foundation
Let $x_1, \dots, x_N \in \mathbb{R}^n$ be the data points. A subset $A \subseteq \mathbb{R}^n$ is called an *affine subspace* if there exist a vector $x \in \mathbb{R}^n$ and a linear subspace $V \subseteq \mathbb{R}^n$ such that
$ A = x + V := \{x + v : v \in V\} $
Its dimension is defined as $\dim(A) := \dim(V)$. 

For an affine subspace $A \subseteq \mathbb{R}^n$, define its *energy* with respect to the data points $x_1, \dots, x_N$ as the sum of squared distances from the points to $A$:
$E(A) := \sum_{i=1}^{N} {dist}(x_i, A)^2$

Let $AG_m$ denote the set of all $m$-dimensional affine subspaces of $\mathbb{R}^n$. An affine subspace $A \in AG_m$ is called a *best-fit* $m$-dimensional affine subspace for $x_1, \dots, x_N$ if it solves the optimization problem
$\min_{B \in AG_m} E(B)$

Finding such a subspace is exactly the goal of PCA.

---
## 1. Reduction to Centered Data
An affine subspace $A \in AG_m$ can always be written as $A = \bar{x} + V$, where $V \in \mathcal{G}_m$ is an $m$-dimensional linear subspace (subspace passing through the origin) and x is the empirical mean of the dataset:
$ x = \frac{1}{N} \sum_{i=1}^{N} x_i $

By centering our dataset ($\tilde{x}_i = x_i - \bar{x}$), the problem of finding the optimal affine subspace simplifies to finding an optimal $m$-dimensional **linear subspace** $V \subseteq \mathbb{R}^n$ for the centered data matrix $X \in \mathbb{R}^{N \times n}$.

---
## 2. Equivalent Formulations: Error Minimization vs. Variance Maximization
For any $m$-dimensional linear subspace $V$ with an orthonormal basis $\{v_1, \dots, v_m\}$, the reconstruction energy $E(V)$ satisfies:
$ E(V) = \sum_{i=1}^{N} \|x_i\|^2 - \sum_{k=1}^{m} \|X v_k\|^2 $

Thus, **minimizing the reconstruction error** is mathematically equivalent to **maximizing the variance** of the projected data points onto the basis vectors:
$ \min_{V \in \mathcal{G}_m} E(V) \iff \max_{\{v_1, \dots, v_m\}} \sum_{k=1}^{m} \|X v_k\|^2 \quad \text{s.t. } \langle v_i, v_j \rangle = \delta_{ij} $

---
## 3. Connection to Singular Value Decomposition (SVD)
Using the SVD of the centered data matrix $X \in \mathbb{R}^{N \times n}$:
$ X = U \Sigma V^T $
where:
- $V = [v_1, v_2, \dots, v_n]$ contains the right singular vectors (the principal directions).
- $\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)$ contains the singular values in descending order $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.

The optimal $m$-dimensional linear subspace $V_m$ is spanned by the first $m$ right singular vectors:
$ V_m = \operatorname{span}(v_1, \dots, v_m) $

### Reconstruction Error & Explained Variance
The minimum error achieved by the optimal $m$-dimensional subspace is given by:
$ E(V_m) = \|X - \tilde{X}\|_F^2 = \sum_{k=m+1}^{r} \sigma_k^2 $
where $\|\cdot\|_F$ is the Frobenius norm.

The fraction of total variance retained by choosing $m$ components is:
$$ \text{Explained Variance Ratio} = \frac{\sum_{k=1}^{m} \sigma_k^2}{\sum_{k=1}^{r} \sigma_k^2} $$
##  Problem Overview
The first algo that is 'from scratch' implemented, works on Boston Housing Dataset set from https://www.statlearning.com/resources-python

Given data points $x_1, \dots, x_N \in \mathbb{R}^8$ representing 8 selected features from the Boston Housing Dataset (`crim`, `nox`, `rm`, `age`, `dis`, `tax`, `ptratio`, `lstat`), we construct:

- A support vector (Stützvektor) $a \in \mathbb{R}^8$
- An orthonormal basis $\{v_1, v_2, v_3\} \subset \mathbb{R}^8$

Such that $a + \text{Span}(\{v_1, v_2, v_3\})$ forms a **3-best-fitting affine subspace** for the data points.

---


## First approach 
is using **constrained numerical optimization** (`scipy.optimize.minimize`) rather than standard Singular Value Decomposition (SVD) 
### 1. Data Matrix & Centering
Construct the data matrix $X \in \mathbb{R}^{N \times 8}$ and compute the empirical mean:

$a = \bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \quad X_0 = X - a$

### 2. First Principal Component ($v_1$)
Solve the optimization problem:

$\max { \Vert{}X_0 v\Vert{}^2 : v \in \mathbb{R}^8, \Vert{}v\Vert{}=1 }$

### 3. Subsequent Components ($v_m$ for $m \ge 2$)
Iteratively solve for each $v_m$:

$\max { \Vert{}X_0 v\Vert{}^2 : v \in \mathbb{R}^8, \Vert{}v\Vert{}=1, \langle v, v_j \rangle = 0 \quad \forall j = 1, \dots, m-1 }$

Adding orthogonality constraints relative to all previously found basis vectors.



