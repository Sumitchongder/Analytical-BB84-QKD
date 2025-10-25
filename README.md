# 🔐 Analytical BB84 Engine  

#### **Team Name:** AnalytiQ  
#### **Team Member Name:** Sumit Tapas Chongder [M25IQT013]
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
🖼️ *Caption:* “Quantum circuit representation of BB84: Alice prepares qubits, Eve intercepts, Bob measures”


---

## 🛠️ Implementation

### 3. Workflow Flowchart
🖼️ *Caption:* “BB84 protocol workflow: from qubit preparation to key distillation”

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

### Error Checking: QBER + Decoy Yields
- We perform error checking by estimating QBER from sifted bits and validating decoy state yields to ensure channel integrity and detect eavesdropping. 
- The Expected Order should be: Signal > Decoy > Vacuum. Deviations can indicate eavesdropping or channel anomalies. And, the QBER value should be less than the Abort threshold (QBER) value set otherwise, QBER aborts.



### Comparative Visualizations: 
1. QBER vs Intercept Probability

   
2. QBER vs Depolarizing Noise

   
3. QBER vs Probabilistic Skew

   


### QBER Heatmap with Threshold Contour 
🖼️ *Caption:* “QBER heatmap showing Secure and Abort zones with cyan threshold contour”


- QBER increases with intercept probability and channel noise  
- Cyan contour marks threshold boundary  
- Secure vs Abort zones clearly annotated  
- Decoy yields follow expected ordering  
- Analytical mode enables instant heatmap generation

---

## 💡 Innovation

### 5. Error Correction + Privacy Amplification Using Vectorized QKD Simualtion  
- ✅ Threshold-based modeling: We simulate EC and PA using a configurable QBER threshold (e.g., 11%) to decide whether the protocol proceeds or aborts.
- 🧮 Error Correction (EC): If QBER is below the threshold, we assume classical error correction succeeds and mismatched bits are reconciled.
- 🔐 Privacy Amplification (PA): We reduce the final key length based on estimated information leakage.
- ⚡ Vectorized simulation: This approach avoids slow bit-level operations and enables fast, scalable analysis across many parameter settings.
- 📊 Real-time feedback: Users instantly see whether EC/PA succeeds and how much usable key material remains.
- 🎓 Educational clarity: Makes the final stages of BB84 transparent and easy to understand, reinforcing the role of QBER in quantum security.



### 6. Eve Strategy Comparison  
🖼️ *Caption:* “Side-by-side heatmaps for Intercept-Resend and Probabilistic Skew strategies”

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

## Install miniconda
##### Windows Command Prompt
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe \
start /wait "" .\miniconda.exe /S \
del .\miniconda.exe

##### macOS
mkdir -p ~/miniconda3 \
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
rm ~/miniconda3/miniconda.sh 

##### Linux
https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

In case of any future updates, you can check the following link for qiskit installation.\
https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit

## Install environment
Download qcl.yml \
Open Anaconda prompt (windows) \
Open terminal (macOS and Linux)

#### Change the directory where the qcl.yml is downloaded

#### conda env create -f qcl.yml 

<img width="2308" height="348" alt="Image" src="https://github.com/user attachments/assets/d5f2c00c-f1f4-4767-89d0-55884e554806" />

## Activate Environment and Run the Streamlit.py Application
Activate the qcl Environment by using the following command:
##### conda activate qcl

Change the directory where you have downloaded the Python File (streamlit_app.py)
For Example:
##### cd Downloads {if the python (.py) file in Downloads Folder}

To run the streamlit_app.py execute the following command
#### streamlit run streamlit_app.py

<img width="2316" height="557" alt="Image" src="https://github.com/user-attachments/assets/da77ed62-6ec7-4d9e-a81f-308e5bc041e0" />


