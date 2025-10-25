# 🔐 Analytical BB84 Engine  

#### **Team Name:** AnalytiQ  
#### **Hackathon:** Qiskit Fall Fest 2025  
#### **Project:** Analytical Acceleration of the BB84 Protocol — An Interactive Framework for Quantum Key Distribution Security Analysis

---

## 📖 Overview

This project presents a complete, interactive implementation of the BB84 Quantum Key Distribution (QKD) protocol using **Qiskit** and **Streamlit**. It demonstrates how two parties (Alice and Bob) can securely generate a shared key while detecting any eavesdropping attempts by Eve. The system features a dual-backend architecture:

- **Simulator Mode** using Qiskit Aer for correctness validation  
- **Analytical Turbo Mode** for scalable, real-time visualizations of Quantum Bit Error Rate (QBER)

Our app includes strategy-aware dashboards, annotated heatmaps, and secure/abort zone visualizations — making quantum security fast, clear, and interactive.

---

## 🚀 Features

- ✅ Full BB84 protocol workflow: Alice → Eve → Bob → Sifting → QBER → EC/PA  
- ⚡ Dual execution modes: Simulator (Qiskit Aer) and Analytical Turbo Mode
- 🧠 Eavesdropper simulation: Intercept-Resend and Probabilistic Skew strategies  
- 📊 Security phase diagrams: QBER heatmaps with cyan threshold contours  
- 🔍 Decoy state analysis: Signal > Decoy > Vacuum yield ordering  
- 🧪 Noise modeling: Depolarizing channel with adjustable parameters  
- 🧰 Modular codebase with caching and reproducibility  
- 🎓 Educational clarity with annotated visuals and interactive UI

---
### 1. Streamlit App Homepage 
🖼️ *Caption:* “Streamlit dashboard for BB84 protocol execution and visualization”

---

## 🧠 Why BB84?

As quantum computers advance, classical encryption methods like RSA and ECC face potential compromise. BB84 offers a physics-based alternative: any attempt to intercept qubits introduces detectable errors. This project makes BB84 tangible through simulation, visualization, and analysis.

---

### 2. BB84 Circuit Diagram  


---

## 🛠️ Implementation

### 🔧 Technologies Used
- Python 3.10+  
- Qiskit  
- Streamlit  
- NumPy  
- Matplotlib / Seaborn

---

### 🧪 Protocol Steps
1. Alice generates random bits and bases  
2. Qubits are prepared in Z or X basis  
3. Eve intercepts and measures (optional)  
4. Bob measures with random bases  
5. Sifting: matched bases retained  
6. QBER estimated from revealed subset  
7. If QBER ≤ threshold → EC/PA → secure key  
8. If QBER > threshold → protocol aborts

---

## 📈 Results

- QBER increases with intercept probability and channel noise  
- Cyan contour marks threshold boundary  
- Secure vs Abort zones clearly annotated  
- Decoy yields follow expected ordering  
- Analytical mode enables instant heatmap generation

---

## 💡 Innovation

- **Turbo Mode:** Analytical backend with vectorized probability model  
- **Strategy-Aware Dashboards:** Compare Eve strategies side-by-side  
- **Secure/Abort Zone Visualization:** Cyan contours + region labels  
- **Caching & Reproducibility:** Instant demo experience for judges  
- **Educational Design:** Clear labels, color scales, and annotations

---

## 🎓 Educational Value

This app is designed to teach QKD interactively:
- Visualizes quantum disturbance and eavesdropping detection  
- Compares QKD to post-quantum cryptography (PQC)  
- Ideal for students, researchers, and hackathon judges

---

## 📦 Installation and Execution 




