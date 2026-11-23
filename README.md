# Custom PCA via Constrained Optimization (Boston Housing Dataset)

This repository provides a step-by-step implementation of **Principal Component Analysis (PCA)** using **constrained numerical optimization** (`scipy.optimize.minimize`) rather than standard Singular Value Decomposition (SVD). 

The goal is to find an optimal $m$-dimensional affine subspace for high-dimensional statistical data.

---

## 📌 Problem Overview

Given data points $x_1, \dots, x_N \in \mathbb{R}^8$ representing 8 selected features from the Boston Housing Dataset (`crim`, `nox`, `rm`, `age`, `dis`, `tax`, `ptratio`, `lstat`), we construct:

- A support vector (Stützvektor) $a \in \mathbb{R}^8$
- An orthonormal basis $\{v_1, v_2, v_3\} \subset \mathbb{R}^8$

Such that $a + \text{Span}(\{v_1, v_2, v_3\})$ forms a **3-best-fitting affine subspace** for the data points.

---

## 📐 Mathematical Formulation (Algorithm 2.2.1)

### 1. Data Matrix & Centering
Construct the data matrix $X \in \mathbb{R}^{N \times 8}$ and compute the empirical mean:

$$a = \bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \quad X_0 = X - a$$

### 2. First Principal Component ($v_1$)
Solve the optimization problem:

$$\max { \Vert{}X_0 v\Vert{}^2 : v \in \mathbb{R}^8, \Vert{}v\Vert{}=1 }$$

*(In code, we minimize $-\Vert{}X_0 v\Vert{}^2$ under the equality constraint $\Vert{}v\Vert{}^2 - 1 = 0$.)*

### 3. Subsequent Components ($v_m$ for $m \ge 2$)
Iteratively solve for each $v_m$:

$$\max { \Vert{}X_0 v\Vert{}^2 : v \in \mathbb{R}^8, \Vert{}v\Vert{}=1, \langle v, v_j \rangle = 0 \quad \forall j = 1, \dots, m-1 }$$

Adding orthogonality constraints relative to all previously found basis vectors.



