# Physics-Consistent Radar Emulation (WGAN-GP)

![Project Status](https://img.shields.io/badge/Status-Research_Prototype-blue) 
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![MATLAB](https://img.shields.io/badge/MATLAB-Phased_Array_Toolbox-orange) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Statistical Emulation of Single-CPI Radar Range-Doppler Maps using Conditional WGAN-GP**

## 📖 1. Purpose and Context
This project addresses the challenge of data scarcity in radar research by combining **physics-based simulation** with **generative AI**. The primary objective is to investigate whether a **Conditional Wasserstein GAN (cWGAN-GP)** can statistically emulate intermediate radar processing products—specifically single-CPI Range-Doppler (RD) maps—when trained on physics-consistent synthetic data.

The work serves as a methodological exploration into using generative models as nonlinear density estimators for complex radar signatures.

---

## 📡 2. Radar Signal Processing Background

### 2.1 Pulse-Doppler Physics
The ground truth data is generated using a high-fidelity simulation in **MATLAB's Phased Array System Toolbox**. The received baseband signal for a pulse-Doppler radar is modeled as:

$$r(t,n) = \alpha s(t-\tau)e^{j2\pi f_D n T_r} + w(t,n)$$

Where:
* $t$ is fast time (range dimension)
* $n$ is the pulse index (slow time)
* $\tau = 2R/c$ is the round-trip delay
* $f_D = 2v_r f_c / c$ is the Doppler frequency
* $\alpha$ incorporates Radar Cross Section (RCS) and propagation effects
* $w(t,n)$ represents complex thermal noise and clutter

### 2.2 Range-Doppler Processing
The raw signal undergoes **Matched Filtering** (range compression) and **Doppler Processing** (FFT across pulses). The resulting Range-Doppler map is defined as:

$$RD(r, f_D) = |\mathcal{F}_n \{ \int r(t,n) s^*(t-\tau) dt \}|$$

> **Critical Distinction:** This project generates **Single-CPI Range-Doppler maps**, which capture mean Doppler and spread. It does *not* generate micro-Doppler spectrograms, which would require sliding-window time-frequency analysis (STFT).

---

## 🛠️ 3. Data Generation Pipeline

### 3.1 System Parameters
The radar parameters are selected to reflect realistic X-band operation:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Carrier Frequency ($f_c$)** | 10 GHz | X-Band |
| **PRF** | 20 kHz | Pulse Repetition Frequency |
| **Pulse Width** | $1~\mu s$ | Waveform duration |
| **Pulses per CPI ($N_p$)** | 128 | Coherent Processing Interval |

### 3.2 Target & Micro-Motion Modeling
The simulation distinguishes between target classes based on their motion dynamics:
1.  **Jet Aircraft (F-16):** High radial velocity, specific RCS statistics.
2.  **Propeller Aircraft (C-130):** Modeled with specific micro-motion effects introduced as slow-time phase modulation:

$$\phi(n) = 2\pi f_D n T_r + \sum_{k} \beta_k \sin(2\pi f_k n T_r)$$

### 3.3 Clutter & Noise
* **Thermal Noise:** Modeled as complex Gaussian white noise.
* **Clutter:** Modeled as low-pass filtered slow-time clutter to emulate correlated background returns. No post-hoc visual smoothing is applied; the maps retain realistic FFT binning and noise granularity.

---

## 🧠 4. Generative Modeling (cWGAN-GP)

### 4.1 Architecture
The project employs a **Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)**.
* **Generator ($G$):** $G(z, y) \rightarrow \hat{x}$
* **Discriminator ($D$):** $D(x, y) \rightarrow \mathbb{R}$
* **Conditioning:** The model is conditioned on the target label $y$ (Jet/Propeller) to learn class-specific distributions.

### 4.2 Objective Function
Standard GANs suffer from instability. This implementation uses the **Wasserstein-1 distance** with a **Gradient Penalty** term to enforce the Lipschitz constraint:

$$L = \underbrace{\mathbb{E}[D(x)] - \mathbb{E}[D(G(z))]}_{\text{Wasserstein Estimate}} + \underbrace{\lambda \mathbb{E}_{\hat{x}} (||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2}_{\text{Gradient Penalty}}$$

This ensures the generator approximates $p_{GAN}(RD) \approx p_{Radar}(RD)$ with stable convergence.

---

## 📊 5. Results & Validation

The generative model was evaluated against the physics-based ground truth.

* **Visual Fidelity:** The synthetic RD maps preserve Doppler localization, noise statistics, and class-dependent characteristics (e.g., the spread of propeller modulation vs. the focused energy of a jet).
* **Physics Check:** A comparison of pixel intensity distributions confirms that the GAN correctly learns the energy distribution of the radar returns.

*(See `results/` folder for generated comparisons)*

---

## ⚠️ 6. Limitations

* **No Raw IQ Generation:** The model generates processed Range-Doppler maps (magnitude), not the raw complex IQ time-series data.
* **Single-CPI Only:** The model does not capture temporal evolution across multiple CPIs (tracking dynamics).
* **Validation Scope:** Validation is performed against high-fidelity physics simulations, not measured data from a real-world radar deployment.
* **Statistical Emulation:** The GAN is a statistical emulator; it does not solve Maxwell's equations or calculate electromagnetic propagation directly.

---

## 🚀 7. Usage

### Prerequisites
* MATLAB (with Phased Array System Toolbox)
* Python 3.x (Requires NumPy, Matplotlib, and Deep Learning Framework)

### Running the Project
1.  **Generate Data:** Run `generate_dataset.m` in MATLAB to create the training `.mat` files.
2.  **Train Model:** Execute the `cWGAN-GP.ipynb` notebook to train the generator and discriminator.
3.  **Inference:** Use the saved generator weights to synthesize new RD maps conditioned on your target class of choice.

---

## 📜 8. Citation
If you use this code or methodology in your research, please cite this repository as follows:

```bibtex
@software{Dubey_Radar_WGAN_2025,
  author = {Dubey, Manan},
  title = {Physics-Consistent Radar Emulation using Conditional WGAN-GP},
  year = {2025},
  url = {[https://github.com/yourusername/physics-radar-wgan](https://github.com/yourusername/physics-radar-wgan)}
}
