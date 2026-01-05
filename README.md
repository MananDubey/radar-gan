# Generative Emulation of Physics-Based Radar Range-Doppler Maps

![Project Status](https://img.shields.io/badge/Status-Research_Prototype-blue)  [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  ![MATLAB](https://img.shields.io/badge/MATLAB-Phased_Array_Toolbox-orange)  ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Statistical Emulation of Single-CPI Radar Log-Magnitude Range-Doppler Maps using Conditional WGAN-GP**

---

## 📖 1. Purpose and Context

This project addresses data scarcity in radar research by combining **physics-consistent radar simulation** with **generative deep learning**.  
The objective is to evaluate whether a **Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)** can act as a **high-fidelity statistical emulator** for intermediate radar signal processing products—specifically **single-CPI log-magnitude Range-Doppler (RD) maps**.

### ⚠️ Important Distinction

- **The Data:** Generated using a **physics-based MATLAB simulation** (Phased Array System Toolbox), explicitly modeling:
  - X-band propagation
  - Target kinematics
  - Radar cross-section (RCS)
  - Micro-motion induced phase and amplitude modulation
- **The Model:** A **purely data-driven statistical model**.  
  The GAN does **not** solve Maxwell’s equations or enforce physical constraints; it learns the **high-dimensional probability distribution** of physics-generated RD maps.

The goal is to evaluate whether such a model can serve as a **computationally efficient surrogate** for expensive radar simulations while preserving key signal characteristics.

---

## 📡 2. Radar Signal Processing Background

### 2.1 Pulse-Doppler Signal Model

The received complex baseband signal for a pulse-Doppler radar is modeled as:

$$
r(t,n) = \alpha\, s(t-\tau)\, e^{j2\pi f_D n T_r} + w(t,n)
$$

Where:
- $t$ is fast time (range dimension)
- $n$ is the pulse index (slow time)
- $\tau = 2R/c$ is the round-trip delay
- $f_D = \frac{2 v_r f_c}{c}$ is the Doppler frequency
- $\alpha$ accounts for RCS and propagation losses
- $w(t,n)$ represents complex thermal noise and clutter

---

### 2.2 Range-Doppler Processing

The received signal undergoes:
1. **Matched filtering** (range compression)
2. **FFT across pulses** (Doppler processing)

The resulting Range-Doppler response is:

$$
RD(r, f_D) =
| \mathcal{F}_n \{ \int r(t,n)\, s^*(t-\tau)\, dt \} |
$$

The dataset consists of **single-CPI, log-magnitude (dB) Range-Doppler maps**.

> **Note:**  
> This work does **not** generate micro-Doppler spectrograms.  
> Micro-motion effects appear only as Doppler spread within a single CPI.

---

## 🛠️ 3. Data Generation Pipeline

### 3.1 Radar System Parameters

| Parameter | Value | Description |
|--------|------|-------------|
| Carrier Frequency ($f_c$) | 10 GHz | X-band |
| Sample Rate | 1 MHz | Fast-time sampling |
| PRF | 20 kHz | Pulse repetition frequency |
| Pulse Width | 1 µs | Rectangular waveform |
| Pulses per CPI | 128 | Single CPI |
| Doppler FFT Length | 256 | Zero-padded FFT |

---

### 3.2 Target Classes & Motion Modeling

Targets are grouped into **two motion classes**, each containing multiple platforms:

**Jet Aircraft (Class 1)**  
Examples: F-16, F-15, Su-57  
- High radial velocity
- Dominant bulk Doppler
- Weak sinusoidal micro-motion phase modulation

**Propeller / Rotorcraft (Class 2)**  
Examples: C-130, MQ-9, UAV  
- Lower bulk velocity
- Blade-induced phase and amplitude modulation
- Multiple rotating scatterers

Micro-motion is introduced as **slow-time phase modulation**:

$$
\phi(n) = 2\pi f_D n T_r + \sum_k \beta_k \sin(2\pi f_k n T_r + \phi_k)
$$

Amplitude modulation is additionally applied for rotating blades to emulate scintillation effects.

---

### 3.3 System-Reference RCS Calibration

A **system-reference calibration** is performed using a known RCS target prior to dataset generation.  
This establishes a consistent amplitude reference and preserves **relative RCS-dependent energy scaling** across all generated samples.

---

### 3.4 Clutter & Noise Modeling

- **Thermal Noise:** Complex Gaussian white noise
- **Clutter:** Complex Gaussian clutter with **slow-time correlation**, implemented via low-pass filtering across pulses

No post-hoc image smoothing or artificial visual enhancement is applied.

---

### 3.5 Log-Magnitude Conversion

After Range-Doppler processing, magnitude responses are converted to **log-scale (dB)**:

$$
RD_{\text{dB}} = 20 \log_{10}(|RD| + \epsilon)
$$

---

### 3.6 Per-Image Robust Normalization

Each Range-Doppler map is normalized **independently** using percentile-based scaling:

- 5th percentile → 0  
- 95th percentile → 1  

This improves robustness to dynamic-range variation while preserving spatial and spectral structure.

Images are resized to **128 × 128**.

---

### 3.7 GAN Input Scaling

Prior to GAN training, normalized RD maps are linearly scaled to:

$$
x_{\text{GAN}} \in [-1, 1]
$$

This scaling is consistent with WGAN training stability requirements.

---

## 🧠 4. Generative Modeling (cWGAN-GP)

### 4.1 Architecture

The project employs a **Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)**:

- **Generator:**  
  $G(z, y) \rightarrow \hat{x}$
- **Critic (Discriminator):**  
  $D(x, y) \rightarrow \mathbb{R}$

Conditioning is performed using **target motion class labels**, provided to both generator and critic.

---

### 4.2 Objective Function

The critic is trained using the Wasserstein-1 distance with gradient penalty:

$$
\mathcal{L} =
\mathbb{E}[D(x, y)] -
\mathbb{E}[D(G(z, y), y)] +
\lambda \mathbb{E}_{\hat{x}}
\left(\|\nabla_{\hat{x}} D(\hat{x}, y)\|_2 - 1\right)^2
$$

This enforces the Lipschitz constraint and stabilizes training.

---

## 📊 5. Results & Validation

Evaluation is performed against physics-generated ground truth.

- **Visual Fidelity:**  
  Generated RD maps preserve Doppler localization, spectral spread, and class-dependent structure.
- **Statistical Consistency:**  
  Pixel-intensity distributions of generated samples closely match those of the simulated dataset.

Validation is performed against **simulation data only**, not real radar measurements.

---

## ⚠️ 6. Limitations

- **No Raw IQ Generation:** Only log-magnitude RD maps are generated.
- **Single-CPI Only:** No temporal evolution across CPIs.
- **Simulation-Only Validation:** No measured radar data used.
- **Statistical Emulation:** The GAN does not solve electromagnetic propagation.

---

## 🚀 7. Usage

### Prerequisites
- MATLAB (Phased Array System Toolbox)
- Python 3.8+
- TensorFlow **2.8+**
- NumPy, SciPy, Matplotlib

### Workflow
1. Run `generate_dataset.m` to create `.mat` training files.
2. Train the model using `cWGAN-GP.ipynb`.
3. Use the trained generator to synthesize class-conditioned RD maps.

---

## 📜 8. Citation

```bibtex
@software{Dubey_Radar_WGAN_2025,
  author = {Manan Dubey},
  title = {Generative Emulation of Physics-Based Radar Range-Doppler Maps},
  year = {2025},
  url = {https://github.com/MananDubey/radar-gan}
}
