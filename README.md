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
- 🧪 Noise modelling: Depolarizing channel with adjustable parameters  
- 🧰 Modular codebase with caching and reproducibility  
- 🎓 Educational clarity with annotated visuals and interactive UI

---
### 1. Streamlit App Homepage 
🖼️ “Streamlit dashboard for BB84 protocol execution and visualization”

<img width="3839" height="2169" alt="Image" src="https://github.com/user-attachments/assets/8b8c049f-4fb6-499e-9d84-7721a82fcd71" />

<img width="3839" height="2197" alt="Image" src="https://github.com/user-attachments/assets/2d59bfa7-0c35-402c-ad32-0829f07eab6f" />

<img width="3839" height="2145" alt="Image" src="https://github.com/user-attachments/assets/493513f4-bc0a-46ed-867b-75f7af6b9765" />

---

## 🧠 Why BB84?

As quantum computers advance, classical encryption methods like RSA and ECC face potential compromise. BB84 offers a physics-based alternative: any attempt to intercept qubits introduces detectable errors. This project makes BB84 tangible through simulation, visualization, and analysis.

---

### 2. BB84 Circuit Diagram  

##### 🖼️ “Quantum circuit representation of BB84: Alice prepares qubits, Eve does not intercept, Bob measures”

<img width="2241" height="1504" alt="Image" src="https://github.com/user-attachments/assets/e385d86e-a727-4750-be16-8b88eadb6cad" />

##### 🖼️ “Quantum circuit representation of BB84: Alice prepares qubits, Eve intercepts (Intercept Resend Attack), Bob measures”

<img width="3794" height="1888" alt="Image" src="https://github.com/user-attachments/assets/e4bdb1a3-e293-4782-aa11-000bdc6e7343" />

##### 🖼️ “Quantum circuit representation of BB84: Alice prepares qubits, Eve intercepts (Probabilistic Skew Attack), Bob measures”

<img width="3791" height="1912" alt="Image" src="https://github.com/user-attachments/assets/de2f908b-c564-449e-916f-563936cc9dd9" />

---

## 🛠️ Implementation

### 3. Workflow Flowchart

##### 🖼️ “BB84 protocol workflow: from qubit preparation to key distillation”

<img width="1028" height="3118" alt="Image" src="https://github.com/user-attachments/assets/1e43f16c-7a2b-403a-8251-f507be8ab6f4" />

### 🔧 Technologies Used
- Python 3.10+  
- Qiskit  
- Streamlit  
- NumPy  
- Matplotlib / Seaborn

---

### 🧪 Protocol Steps

1. Alice generates random bits and bases  
2. Qubits are prepared in the Z or X basis  
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
- The Expected Order should be: Signal > Decoy > Vacuum. Deviations can indicate eavesdropping or channel anomalies. Additionally, the QBER value should be less than the Abort threshold (QBER) value; otherwise, QBER aborts.

<img width="1645" height="1074" alt="Image" src="https://github.com/user-attachments/assets/a6614bfe-338f-4b2e-82d4-dcd4c4983b41" />


### Comparative Visualizations: 

#### 1. QBER vs Intercept Probability
<img width="1647" height="1787" alt="Image" src="https://github.com/user-attachments/assets/9b03ee15-6820-4af4-9b65-c871136b2689" />
   
#### 2. QBER vs Depolarising Noise
<img width="1636" height="1868" alt="Image" src="https://github.com/user-attachments/assets/75eb4097-8d29-4f34-92af-0598e5fc3fcb" />
   
#### 3. QBER vs Probabilistic Skew
<img width="1631" height="1791" alt="Image" src="https://github.com/user-attachments/assets/0dd1d238-dac8-480e-8786-316927993655" />

   
### QBER Heatmap with Threshold Contour 

🖼️ “QBER heatmap showing Secure and Abort zones with cyan threshold contour (Intercept Resend Attack)”

<img width="1619" height="1954" alt="Image" src="https://github.com/user-attachments/assets/e89120b2-c00b-4f89-9216-221b39df889d" />


🖼️ “QBER heatmap showing Secure and Abort zones with cyan threshold contour (Probability Skew Attack)”

<img width="1612" height="1960" alt="Image" src="https://github.com/user-attachments/assets/416911b0-b5b4-403e-8d9f-9f92576325e0" />

#### Insights: 
- QBER increases with intercept probability and channel noise  
- Cyan contour marks threshold boundary  
- Secure vs Abort zones clearly annotated  
- Decoy yields follow the expected ordering  
- Analytical mode enables instant heatmap generation

---

## 💡 Innovation

### 5. Error Correction + Privacy Amplification (Using Vectorized QKD Simulation)  

<img width="1167" height="654" alt="Image" src="https://github.com/user-attachments/assets/6af31263-6d4b-4ad0-995a-42643911b0d4" />

- ✅ Threshold-based modelling: We simulate EC and PA using a configurable QBER threshold (e.g., 11%) to decide whether the protocol proceeds or aborts.
- 🧮 Error Correction (EC): If QBER is below the threshold, we assume classical error correction succeeds and mismatched bits are reconciled.
- 🔐 Privacy Amplification (PA): We reduce the final key length based on estimated information leakage.
- ⚡ Vectorized simulation: This approach avoids slow bit-level operations and enables fast, scalable analysis across many parameter settings.
- 📊 Real-time feedback: Users instantly see whether EC/PA succeeds and how much usable key material remains.
- 🎓 Educational clarity: Makes the final stages of BB84 transparent and easy to understand, reinforcing the role of QBER in quantum security.


### 6. Eve Strategy Comparison  

🖼️ “Side-by-side heatmaps for Intercept-Resend and Probabilistic Skew strategies”

<img width="2876" height="1170" alt="Image" src="https://github.com/user-attachments/assets/b99ba3fe-3a23-489a-acbe-b28567efc55f" />

- **Turbo Mode:** Analytical backend with vectorized probability model  
- **Strategy-Aware Dashboards:** Compare Eve strategies side-by-side  
- **Secure/Abort Zone Visualization:** Cyan contours + region labels  
- **Caching & Reproducibility:** Instant demo experience for judges  
- **Educational Design:** Clear labels, colour scales, and annotations

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

In case of any future updates, you can check the following link for Qiskit installation.\
https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit

## Install environment
Download qcl.yml \
Open Anaconda prompt (Windows) \
Open terminal (macOS and Linux)

#### Change the directory where the qcl.yml is downloaded

#### conda env create -f qcl.yml 

<img width="2308" height="348" alt="Image" src="https://github.com/user-attachments/assets/5e3f241c-1d0c-4015-88f8-3c7f33c4c0bc" />

## Activate Environment and Run the Streamlit.py Application
Activate the qcl Environment by using the following command:
##### conda activate qcl

Change the directory where you have downloaded the Python File (streamlit_app.py)
For Example:
##### cd Downloads {if the python (.py) file in the Downloads Folder}

To run the streamlit_app.py, execute the following command
#### streamlit run streamlit_app.py

<img width="2316" height="557" alt="Image" src="https://github.com/user-attachments/assets/da77ed62-6ec7-4d9e-a81f-308e5bc041e0" />


