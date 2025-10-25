import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bitarray import bitarray
import random

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

# ======================================================================
# Global setup
# ======================================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
sns.set(context="talk", style="whitegrid", palette="deep")

sim_backend = AerSimulator(method="automatic", seed_simulator=GLOBAL_SEED)

# ======================================================================
# Core BB84 helpers
# ======================================================================
def prepare_bb84_state(bit: int, basis: int) -> QuantumCircuit:
    qc = QuantumCircuit(1)
    if basis == 0:  # Z basis
        if bit == 1:
            qc.x(0)
    else:  # X basis
        if bit == 1:
            qc.x(0)
        qc.h(0)
    return qc

def measure_in_basis(qc: QuantumCircuit, basis: int) -> QuantumCircuit:
    meas = qc.copy()
    if basis == 1:
        meas.h(0)
    meas.measure_all()
    return meas

def build_noise_model(depol_1q_prob=0.0, bitflip_prob=0.0, phaseflip_prob=0.0, readout_error_prob=0.0):
    nm = NoiseModel()
    if depol_1q_prob > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(depol_1q_prob, 1), ["id","x","h","measure"])
    if bitflip_prob > 0:
        nm.add_all_qubit_quantum_error(pauli_error([("X", bitflip_prob), ("I", 1-bitflip_prob)]), ["id","x","h"])
    if phaseflip_prob > 0:
        nm.add_all_qubit_quantum_error(pauli_error([("Z", phaseflip_prob), ("I", 1-phaseflip_prob)]), ["id","x","h"])
    if readout_error_prob > 0:
        nm.add_all_qubit_quantum_error(pauli_error([("X", readout_error_prob), ("I", 1-readout_error_prob)]), ["measure"])
    return nm

def random_bits(n): return np.random.randint(0, 2, size=n)
def random_bases(n, bias=0.5): return (np.random.rand(n) >= (1 - bias)).astype(int)

def sift(alice_bases, bob_bases, alice_bits, bob_results):
    mask = (alice_bases == bob_bases)
    pos = np.where(mask)[0]
    return alice_bits[mask], bob_results[mask], pos

def estimate_qber(alice_sift, bob_sift, sample_fraction=0.25):
    m = len(alice_sift)
    if m == 0:
        return 1.0, [], 0
    sample_size = max(1, int(sample_fraction * m))
    idx = np.random.choice(m, size=sample_size, replace=False)
    agree = np.sum(alice_sift[idx] == bob_sift[idx])
    qber = 1 - (agree / sample_size)
    return qber, idx.tolist(), sample_size

# ======================================================================
# Eve strategies
# ======================================================================
class EveStrategy:
    def __init__(self, intercept_prob=1.0, basis_bias=0.5):
        self.intercept_prob = intercept_prob
        self.basis_bias = basis_bias
    def apply(self, qc: QuantumCircuit) -> QuantumCircuit:
        return qc

class EveInterceptResend(EveStrategy):
    def apply(self, qc: QuantumCircuit) -> QuantumCircuit:
        if np.random.rand() > self.intercept_prob:
            return qc
        eve_basis = 0 if np.random.rand() < self.basis_bias else 1
        eve_meas = qc.copy()
        if eve_basis == 1:
            eve_meas.h(0)
        eve_meas.measure_all()
        counts = sim_backend.run(eve_meas).result().get_counts()
        outcomes, cts = zip(*counts.items())
        probs = np.array(cts) / sum(cts)
        eve_bit = int(np.random.choice(outcomes, p=probs))
        new_qc = QuantumCircuit(1)
        if eve_basis == 0:
            if eve_bit == 1: new_qc.x(0)
        else:
            if eve_bit == 1: new_qc.x(0)
            new_qc.h(0)
        return new_qc

class EveProbabilisticSkew(EveStrategy):
    def apply(self, qc: QuantumCircuit) -> QuantumCircuit:
        if np.random.rand() > self.intercept_prob:
            return qc
        eve_basis = 0 if np.random.rand() < self.basis_bias else 1
        eve_meas = qc.copy()
        if eve_basis == 1:
            eve_meas.h(0)
        eve_meas.measure_all()
        counts = sim_backend.run(eve_meas).result().get_counts()
        outcomes, cts = zip(*counts.items())
        probs = np.array(cts) / sum(cts)
        eve_bit = int(np.random.choice(outcomes, p=probs))
        new_qc = QuantumCircuit(1)
        if eve_basis == 0:
            if eve_bit == 1: new_qc.x(0)
        else:
            if eve_bit == 1: new_qc.x(0)
            new_qc.h(0)
        return new_qc

class EveDoNothing(EveStrategy):
    def apply(self, qc: QuantumCircuit) -> QuantumCircuit:
        return qc

# ======================================================================
# Error correction + Privacy amplification (demo)
# ======================================================================
def toeplitz_hash(key_bits: np.ndarray, out_len: int, seed=GLOBAL_SEED) -> bitarray:
    rng = np.random.default_rng(seed)
    n = len(key_bits)
    v = rng.integers(0, 2, size=n + out_len - 1)
    out = bitarray(out_len)
    for i in range(out_len):
        out[i] = int(np.sum(key_bits * v[i:i+n]) % 2)
    return out

def privacy_amplification(alice_sift, bob_sift, qber, security_margin=0.2):
    m = len(alice_sift)
    if m == 0:
        return bitarray(), bitarray(), 0
    out_len = max(0, int((1 - qber - security_margin) * m))
    alice_key = toeplitz_hash(alice_sift, out_len, GLOBAL_SEED)
    bob_key   = toeplitz_hash(bob_sift,   out_len, GLOBAL_SEED)
    return alice_key, bob_key, out_len

def error_correction_stub(alice_sift, bob_sift, qber, efficiency=1.1):
    if len(alice_sift) == 0:
        return alice_sift, bob_sift, 0
    corrected_len = int(len(alice_sift) / efficiency)
    corrected_bob = alice_sift.copy()
    return alice_sift[:corrected_len], corrected_bob[:corrected_len], corrected_len

def full_pipeline_with_ec(alice_sift, bob_sift, qber, qber_threshold=0.11):
    if qber > qber_threshold:
        return bitarray(), bitarray(), 0, True
    alice_corr, bob_corr, _ = error_correction_stub(alice_sift, bob_sift, qber)
    alice_key, bob_key, out_len = privacy_amplification(alice_corr, bob_corr, qber)
    return alice_key, bob_key, out_len, False

# ======================================================================
# Decoy states
# ======================================================================
def generate_intensity_labels(n, signal_fraction=0.7, decoy_fraction=0.3, vacuum_fraction=0.0):
    total = signal_fraction + decoy_fraction + vacuum_fraction
    if total == 0:
        total = 1.0
        signal_fraction = 1.0
        decoy_fraction = 0.0
        vacuum_fraction = 0.0
    return np.random.choice(
        [2, 1, 0], size=n,
        p=[signal_fraction/total, decoy_fraction/total, vacuum_fraction/total]
    )

def apply_intensity_effect(label: int):
    # Additional depolarizing probability by intensity class
    if label == 2:   # signal
        return 0.0
    elif label == 1: # decoy
        return 0.01
    else:            # vacuum
        return 0.03

def decoy_detection_stats(sent_labels, received_clicks):
    df = pd.DataFrame({"label": sent_labels, "click": received_clicks})
    stats = df.groupby("label")["click"].mean()
    return {"Vacuum": stats.get(0, np.nan), "Decoy": stats.get(1, np.nan), "Signal": stats.get(2, np.nan)}

# ======================================================================
# Simulation helpers
# ======================================================================
def simulate_measurement(qc: QuantumCircuit, basis: int, noise_model=None):
    meas_circ = measure_in_basis(qc, basis)
    backend = sim_backend if noise_model is None else AerSimulator(
        method="automatic", noise_model=noise_model, seed_simulator=GLOBAL_SEED
    )
    counts = backend.run(meas_circ).result().get_counts()
    outcomes, cts = zip(*counts.items())
    probs = np.array(cts) / sum(cts)
    return int(np.random.choice(outcomes, p=probs))

# ======================================================================
# Sweep utilities for comparative plots (hybrid approach)
# ======================================================================
@st.cache_data
def sweep_intercept_probs_cached(*args, **kwargs):
    return sweep_intercept_probs(*args, **kwargs)

@st.cache_data
def sweep_noise_levels_cached(*args, **kwargs):
    return sweep_noise_levels(*args, **kwargs)

@st.cache_data
def sweep_probabilistic_skew_cached(*args, **kwargs):
    return sweep_probabilistic_skew(*args, **kwargs)

@st.cache_data
def qber_heatmap_strategy_cached(*args, **kwargs):
    return qber_heatmap_strategy(*args, **kwargs)

# ======================================================================
# Sweep utilities for comparative plots
# ======================================================================
def sweep_intercept_probs(n_bits, eve_cls, depol_1q_prob, eve_basis_bias, basis_bias, sample_fraction, qber_threshold):
    intercept_grid = np.linspace(0, 1.0, 6)
    records = []
    for p in intercept_grid:
        alice_bits = random_bits(n_bits)
        alice_bases = random_bases(n_bits, bias=basis_bias)
        bob_bases = random_bases(n_bits, bias=basis_bias)
        bob_results = np.zeros(n_bits, dtype=int)
        eve = eve_cls(intercept_prob=p, basis_bias=eve_basis_bias)
        for i in range(n_bits):
            qc = prepare_bb84_state(int(alice_bits[i]), int(alice_bases[i]))
            qc_e = eve.apply(qc)
            nm = build_noise_model(depol_1q_prob=depol_1q_prob)
            bob_results[i] = simulate_measurement(qc_e, int(bob_bases[i]), noise_model=nm)
        alice_sift, bob_sift, _ = sift(alice_bases, bob_bases, alice_bits, bob_results)
        qber, _, _ = estimate_qber(alice_sift, bob_sift, sample_fraction)
        _, _, out_len, aborted = full_pipeline_with_ec(alice_sift, bob_sift, qber, qber_threshold)
        records.append({"Intercept Prob": p, "QBER": qber, "Abort": aborted, "Final Key Len": out_len})
    return pd.DataFrame(records)

def sweep_noise_levels(n_bits, eve_cls, intercept_prob, eve_basis_bias, basis_bias, sample_fraction, qber_threshold):
    depol_grid = np.linspace(0, 0.06, 7)
    records = []
    for d in depol_grid:
        alice_bits = random_bits(n_bits)
        alice_bases = random_bases(n_bits, bias=basis_bias)
        bob_bases = random_bases(n_bits, bias=basis_bias)
        bob_results = np.zeros(n_bits, dtype=int)
        eve = eve_cls(intercept_prob=intercept_prob, basis_bias=eve_basis_bias)
        for i in range(n_bits):
            qc = prepare_bb84_state(int(alice_bits[i]), int(alice_bases[i]))
            qc_e = eve.apply(qc)
            nm = build_noise_model(depol_1q_prob=d)
            bob_results[i] = simulate_measurement(qc_e, int(bob_bases[i]), noise_model=nm)
        alice_sift, bob_sift, _ = sift(alice_bases, bob_bases, alice_bits, bob_results)
        qber, _, _ = estimate_qber(alice_sift, bob_sift, sample_fraction)
        _, _, out_len, aborted = full_pipeline_with_ec(alice_sift, bob_sift, qber, qber_threshold)
        records.append({"Depolarizing Noise": d, "QBER": qber, "Abort": aborted, "Final Key Len": out_len})
    return pd.DataFrame(records)

def sweep_probabilistic_skew(n_bits, intercept_prob, eve_basis_bias_range, basis_bias, sample_fraction, qber_threshold):
    records = []
    for bias in eve_basis_bias_range:
        alice_bits = random_bits(n_bits)
        alice_bases = random_bases(n_bits, bias=basis_bias)
        bob_bases = random_bases(n_bits, bias=basis_bias)
        bob_results = np.zeros(n_bits, dtype=int)
        eve = EveProbabilisticSkew(intercept_prob=intercept_prob, basis_bias=bias)
        for i in range(n_bits):
            qc = prepare_bb84_state(int(alice_bits[i]), int(alice_bases[i]))
            qc_e = eve.apply(qc)
            nm = build_noise_model(depol_1q_prob=0.0)
            bob_results[i] = simulate_measurement(qc_e, int(bob_bases[i]), noise_model=nm)
        alice_sift, bob_sift, _ = sift(alice_bases, bob_bases, alice_bits, bob_results)
        qber, _, _ = estimate_qber(alice_sift, bob_sift, sample_fraction)
        _, _, out_len, aborted = full_pipeline_with_ec(alice_sift, bob_sift, qber, qber_threshold)
        records.append({"Eve Basis Bias": bias, "QBER": qber, "Abort": aborted, "Final Key Len": out_len})
    return pd.DataFrame(records)

def tri_panel_dashboard(n_bits, basis_bias, sample_fraction, qber_threshold):
    df_intercept = sweep_intercept_probs(n_bits, EveInterceptResend, 0.0, 0.5,
                                         basis_bias, sample_fraction, qber_threshold)
    df_noise = sweep_noise_levels(n_bits, EveDoNothing, 0.0, 0.5,
                                  basis_bias, sample_fraction, qber_threshold)
    bias_range = np.linspace(0, 1.0, 6)
    df_skew = sweep_probabilistic_skew(n_bits, intercept_prob=1.0,
                                       eve_basis_bias_range=bias_range,
                                       basis_bias=basis_bias,
                                       sample_fraction=sample_fraction,
                                       qber_threshold=qber_threshold)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: InterceptResend
    axes[0].plot(df_intercept["Intercept Prob"], df_intercept["QBER"], marker="o", color="royalblue")
    axes[0].axhline(qber_threshold, color="red", linestyle="--", label="Abort threshold")
    axes[0].fill_between(df_intercept["Intercept Prob"], 0, qber_threshold, color="green", alpha=0.1, label="Secure")
    axes[0].fill_between(df_intercept["Intercept Prob"], qber_threshold, 1, color="red", alpha=0.1, label="Insecure")
    axes[0].set_title("InterceptResend: QBER vs Intercept Prob")
    axes[0].set_xlabel("Intercept Probability"); axes[0].set_ylabel("QBER")
    axes[0].legend()

    # Panel 2: Noise
    axes[1].plot(df_noise["Depolarizing Noise"], df_noise["QBER"], marker="o", color="darkgreen")
    axes[1].axhline(qber_threshold, color="red", linestyle="--", label="Abort threshold")
    axes[1].fill_between(df_noise["Depolarizing Noise"], 0, qber_threshold, color="green", alpha=0.1)
    axes[1].fill_between(df_noise["Depolarizing Noise"], qber_threshold, 1, color="red", alpha=0.1)
    axes[1].set_title("DoNothing: QBER vs Noise")
    axes[1].set_xlabel("Depolarizing Probability"); axes[1].set_ylabel("QBER")
    axes[1].legend()

    # Panel 3: ProbabilisticSkew
    axes[2].plot(df_skew["Eve Basis Bias"], df_skew["QBER"], marker="o", color="purple")
    axes[2].axhline(qber_threshold, color="red", linestyle="--", label="Abort threshold")
    axes[2].fill_between(df_skew["Eve Basis Bias"], 0, qber_threshold, color="green", alpha=0.1)
    axes[2].fill_between(df_skew["Eve Basis Bias"], qber_threshold, 1, color="red", alpha=0.1)
    axes[2].set_title("ProbabilisticSkew: QBER vs Basis Bias")
    axes[2].set_xlabel("Eve Basis Bias"); axes[2].set_ylabel("QBER")
    axes[2].legend()

    plt.tight_layout()
    return fig, df_intercept, df_noise, df_skew

def qber_heatmap_strategy(n_bits, strategy, eve_basis_bias, basis_bias, sample_fraction,
                          depol_range=np.linspace(0, 0.06, 7), intercept_range=np.linspace(0, 1.0, 6)):
    grid = np.zeros((len(depol_range), len(intercept_range)))
    for i, d in enumerate(depol_range):
        for j, p in enumerate(intercept_range):
            alice_bits = random_bits(n_bits)
            alice_bases = random_bases(n_bits, bias=basis_bias)
            bob_bases = random_bases(n_bits, bias=basis_bias)
            bob_results = np.zeros(n_bits, dtype=int)
            eve = strategy(intercept_prob=p, basis_bias=eve_basis_bias)
            for k in range(n_bits):
                qc = prepare_bb84_state(int(alice_bits[k]), int(alice_bases[k]))
                qc_e = eve.apply(qc)
                nm = build_noise_model(depol_1q_prob=d)
                bob_results[k] = simulate_measurement(qc_e, int(bob_bases[k]), noise_model=nm)
            alice_sift, bob_sift, _ = sift(alice_bases, bob_bases, alice_bits, bob_results)
            qber, _, _ = estimate_qber(alice_sift, bob_sift, sample_fraction)
            grid[i, j] = qber
    df = pd.DataFrame(
        grid,
        index=[f"{x:.3f}" for x in depol_range],
        columns=[f"{y:.2f}" for y in intercept_range]
    )
    return df

# ======================================================================
# Streamlit UI
# ======================================================================
st.title("BB84 QKD — Full Workflow with Decoy States, Probabilistic Skew, and Heatmaps")
st.markdown("Run the six-step BB84 workflow, then explore comparative plots, a tri-panel dashboard, and strategy-aware heatmaps.")

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    n_bits = st.slider("Total transmitted bits", 512, 8192, 2048, step=512)
    basis_bias = st.slider("Alice/Bob basis bias (prob Z)", 0.0, 1.0, 0.5, 0.05)
    sample_fraction = st.slider("Parameter estimation fraction", 0.05, 0.5, 0.25, 0.05)
with col2:
    depol_1q_prob = st.slider("Depolarizing error", 0.0, 0.1, 0.01, 0.005)
    bitflip_prob = st.slider("Bit-flip error", 0.0, 0.1, 0.0, 0.005)
    phaseflip_prob = st.slider("Phase-flip error", 0.0, 0.1, 0.0, 0.005)
    readout_error_prob = st.slider("Readout error", 0.0, 0.1, 0.0, 0.005)
with col3:
    intercept_prob = st.slider("Eve intercept probability", 0.0, 1.0, 0.5, 0.05)
    eve_basis_bias = st.slider("Eve basis bias (prob Z)", 0.0, 1.0, 0.5, 0.05)
    qber_threshold = st.slider("Abort threshold (QBER)", 0.05, 0.2, 0.11, 0.005)

eve_strategy_name = st.selectbox("Eve strategy", ["DoNothing", "InterceptResend", "ProbabilisticSkew"])
eve_cls = {"DoNothing": EveDoNothing, "InterceptResend": EveInterceptResend, "ProbabilisticSkew": EveProbabilisticSkew}[eve_strategy_name]

# Decoy states
st.subheader("Decoy states configuration")
colD, colE, colF = st.columns(3)
with colD:
    signal_fraction = st.slider("Signal fraction", 0.0, 1.0, 0.7, 0.05)
with colE:
    decoy_fraction = st.slider("Decoy fraction", 0.0, 1.0, 0.3, 0.05)
with colF:
    vacuum_fraction = st.slider("Vacuum fraction", 0.0, 1.0, 0.0, 0.05)

# Six-step workflow state
if "workflow" not in st.session_state:
    st.session_state.workflow = {}

st.markdown("## Stepwise Workflow")

# Step 1: Alice
if st.button("1) Alice: prepare bits, bases, intensities"):
    alice_bits = random_bits(n_bits)
    alice_bases = random_bases(n_bits, bias=basis_bias)
    intensity_labels = generate_intensity_labels(n_bits, signal_fraction, decoy_fraction, vacuum_fraction)
    st.session_state.workflow.update({"alice_bits": alice_bits, "alice_bases": alice_bases, "intensity_labels": intensity_labels})
    st.success(f"Alice prepared {n_bits} qubits.")

# Step 2: Eve
button_label = f"2) Eve: apply {eve_strategy_name} strategy"
if st.button(button_label):
    if "alice_bits" not in st.session_state.workflow:
        st.error("Run Step 1 first.")
    else:
        eve = eve_cls(intercept_prob=intercept_prob, basis_bias=eve_basis_bias)
        sent_qcs, extra_depol_list = [], []
        for bit, basis, label in zip(st.session_state.workflow["alice_bits"],
                                     st.session_state.workflow["alice_bases"],
                                     st.session_state.workflow["intensity_labels"]):
            qc = prepare_bb84_state(int(bit), int(basis))
            qc_e = eve.apply(qc)
            sent_qcs.append(qc_e)
            extra_depol_list.append(apply_intensity_effect(label))
        st.session_state.workflow.update({"sent_qcs": sent_qcs, "extra_depol_list": extra_depol_list})
        st.success(f"Eve applied {eve_strategy_name} (intercept_prob={intercept_prob}, basis_bias={eve_basis_bias}).")

# Step 3: Bob
if st.button("3) Bob: measure qubits"):
    if "sent_qcs" not in st.session_state.workflow:
        st.error("Run Step 2 first.")
    else:
        bob_bases = random_bases(n_bits, bias=basis_bias)
        bob_results = np.zeros(n_bits, dtype=int)
        clicks = np.zeros(n_bits, dtype=int)
        for i, qc_e in enumerate(st.session_state.workflow["sent_qcs"]):
            nm = build_noise_model(depol_1q_prob=depol_1q_prob + st.session_state.workflow["extra_depol_list"][i],
                                   bitflip_prob=bitflip_prob, phaseflip_prob=phaseflip_prob, readout_error_prob=readout_error_prob)
            bob_results[i] = simulate_measurement(qc_e, int(bob_bases[i]), noise_model=nm)
            label = st.session_state.workflow["intensity_labels"][i]
            # Simple click model per class
            if label == 2:
                clicks[i] = 1 if np.random.rand() < 0.98 else 0
            elif label == 1:
                clicks[i] = 1 if np.random.rand() < 0.90 else 0
            else:
                clicks[i] = 1 if np.random.rand() < 0.10 else 0
        st.session_state.workflow.update({"bob_bases": bob_bases, "bob_results": bob_results, "clicks": clicks})
        st.success("Bob measured all qubits.")

# Step 4: Sifting
if st.button("4) Sifting: keep matched bases"):
    needed = ["alice_bits", "alice_bases", "bob_bases", "bob_results"]
    if not all(k in st.session_state.workflow for k in needed):
        st.error("Run Steps 1–3 first.")
    else:
        alice_sift, bob_sift, positions = sift(
            st.session_state.workflow["alice_bases"],
            st.session_state.workflow["bob_bases"],
            st.session_state.workflow["alice_bits"],
            st.session_state.workflow["bob_results"]
        )
        st.session_state.workflow.update({"alice_sift": alice_sift, "bob_sift": bob_sift, "sift_positions": positions})
        st.success(f"Sifting complete. Kept {len(alice_sift)} of {n_bits} bits.")

# Step 5: Error checking
if st.button("5) Error checking: QBER + decoy yields"):
    if "alice_sift" not in st.session_state.workflow:
        st.error("Run Step 4 first.")
    else:
        qber, idx, sample_size = estimate_qber(st.session_state.workflow["alice_sift"],
                                               st.session_state.workflow["bob_sift"],
                                               sample_fraction)
        st.session_state.workflow.update({"qber": qber, "est_idx": idx, "sample_size": sample_size})
        st.subheader("QBER result")
        st.write({"QBER": round(qber, 4), "Sample size": sample_size})
        stats = decoy_detection_stats(st.session_state.workflow["intensity_labels"], st.session_state.workflow["clicks"])
        st.subheader("Decoy yields (click probability)")
        yield_display = {k: (None if (v is None or np.isnan(v)) else round(float(v), 4)) for k, v in stats.items()}
        st.write(yield_display)
        st.info("Expected order: Signal > Decoy > Vacuum. Deviations can indicate eavesdropping or channel anomalies.")

# --- Optimized QBER estimation (vectorized) ---
def estimate_qber_fast(alice_sift, bob_sift, sample_fraction=0.25):
    """
    Vectorized QBER estimation.
    alice_sift, bob_sift: numpy arrays of sifted bits
    sample_fraction: fraction of bits to reveal for error estimation
    """
    m = len(alice_sift)
    if m == 0:
        return 1.0, [], 0
    sample_size = max(1, int(sample_fraction * m))
    idx = np.random.choice(m, size=sample_size, replace=False)
    qber = np.mean(alice_sift[idx] != bob_sift[idx])
    return qber, idx.tolist(), sample_size
        
# --- Analytical BB84 measurement ---
def measure_bb84_analytical(alice_bit, alice_basis, bob_basis,
                            depol_prob=0.0, bitflip_prob=0.0, phaseflip_prob=0.0):
    """
    Returns Bob's measurement result analytically, without simulator.
    """
    # If bases match: Bob gets Alice's bit with high probability
    if alice_basis == bob_basis:
        result = alice_bit
    else:
        # If bases differ: Bob's result is random
        result = np.random.randint(0, 2)

    # Apply noise analytically
    if np.random.rand() < depol_prob:
        result = np.random.randint(0, 2)  # depolarizing = randomize
    if np.random.rand() < bitflip_prob:
        result ^= 1
    # phaseflip doesn’t affect Z-basis measurement, but if Bob measured in X basis,
    # it flips the outcome
    if bob_basis == 1 and np.random.rand() < phaseflip_prob:
        result ^= 1

    return result


# Step 6: EC + PA (Optimized + Cached)
if st.button("6) Error correction + privacy amplification (Vectorized QKD Simulator)"):
    if "alice_sift" not in st.session_state.workflow:
        st.error("Run Step 5 first.")
    else:
        # Compute QBER using fast vectorized estimator
        qber, idx, sample_size = estimate_qber_fast(
            np.array(st.session_state.workflow["alice_sift"]),
            np.array(st.session_state.workflow["bob_sift"]),
            sample_fraction
        )
        st.session_state.workflow.update({"qber": qber})

        # Run EC + PA
        alice_key, bob_key, out_len, aborted = full_pipeline_with_ec(
            st.session_state.workflow["alice_sift"],
            st.session_state.workflow["bob_sift"],
            qber,
            qber_threshold
        )
        st.session_state.workflow.update({
            "alice_key": alice_key, "bob_key": bob_key,
            "final_key_len": out_len, "aborted": aborted
        })

        st.subheader("Final key summary (Optimized)")
        st.write({
            "QBER": round(qber, 4),
            "Abort": aborted,
            "Final key length": out_len,
            "Keys equal": (alice_key == bob_key) and (out_len > 0)
        })

        # --- Precompute sweeps with caching ---
        st.session_state["df_intercept"] = sweep_intercept_probs_cached(
            n_bits, EveInterceptResend, depol_1q_prob, eve_basis_bias,
            basis_bias, sample_fraction, qber_threshold
        )
        st.session_state["df_noise"] = sweep_noise_levels_cached(
            n_bits, EveDoNothing, intercept_prob, eve_basis_bias,
            basis_bias, sample_fraction, qber_threshold
        )
        bias_range = np.linspace(0, 1.0, 6)
        st.session_state["df_skew"] = sweep_probabilistic_skew_cached(
            n_bits, intercept_prob=1.0, eve_basis_bias_range=bias_range,
            basis_bias=basis_bias, sample_fraction=sample_fraction,
            qber_threshold=qber_threshold
        )
        st.session_state["df_heat_intercept"] = qber_heatmap_strategy_cached(
            n_bits, EveInterceptResend, eve_basis_bias, basis_bias, sample_fraction
        )
        st.session_state["df_heat_skew"] = qber_heatmap_strategy_cached(
            n_bits, EveProbabilisticSkew, eve_basis_bias, basis_bias, sample_fraction
        )

        st.success("All sweeps precomputed and cached. Plots will now render instantly.")
        
# ======================================================================
# Comparative Visualizations
# ======================================================================
st.markdown("---")
st.subheader("Comparative Visualizations")

if st.button("Plot QBER vs Intercept Probability"):
    if "df_intercept" in st.session_state:
        df_plot = st.session_state["df_intercept"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_plot["Intercept Prob"], df_plot["QBER"], marker="o", color="royalblue")
        ax.axhline(qber_threshold, color="red", linestyle="--", label="Abort threshold")
        ax.set_xlabel("Intercept Probability"); ax.set_ylabel("QBER"); ax.legend()
        st.pyplot(fig); st.dataframe(df_plot)
    else:
        st.warning("Run Step 6 first to precompute results.")

if st.button("Plot QBER vs Depolarizing Noise"):
    if "df_noise" in st.session_state:
        df_noise = st.session_state["df_noise"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_noise["Depolarizing Noise"], df_noise["QBER"], marker="o", color="darkgreen")
        ax.axhline(qber_threshold, color="red", linestyle="--", label="Abort threshold")
        ax.set_xlabel("Depolarizing Probability"); ax.set_ylabel("QBER"); ax.legend()
        st.pyplot(fig); st.dataframe(df_noise)
    else:
        st.warning("Run Step 6 first to precompute results.")

if st.button("Plot QBER vs ProbabilisticSkew"):
    if "df_skew" in st.session_state:
        df_skew = st.session_state["df_skew"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_skew["Eve Basis Bias"], df_skew["QBER"], marker="o", color="purple")
        ax.axhline(qber_threshold, color="red", linestyle="--", label="Abort threshold")
        ax.set_xlabel("Eve Basis Bias"); ax.set_ylabel("QBER"); ax.legend()
        st.pyplot(fig); st.dataframe(df_skew)
    else:
        st.warning("Run Step 6 first to precompute results.")
        
# ======================================================================
# Tri-panel Dashboard
# ======================================================================
st.markdown("---")
st.subheader("Tri-panel Dashboard")
if st.button("Show Tri-panel Dashboard"):
    if all(k in st.session_state for k in ["df_intercept","df_noise","df_skew"]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Panel 1
        df_intercept = st.session_state["df_intercept"]
        axes[0].plot(df_intercept["Intercept Prob"], df_intercept["QBER"], marker="o", color="royalblue")
        axes[0].axhline(qber_threshold, color="red", linestyle="--")
        axes[0].set_title("InterceptResend"); axes[0].set_xlabel("Intercept Probability"); axes[0].set_ylabel("QBER")
        # Panel 2
        df_noise = st.session_state["df_noise"]
        axes[1].plot(df_noise["Depolarizing Noise"], df_noise["QBER"], marker="o", color="darkgreen")
        axes[1].axhline(qber_threshold, color="red", linestyle="--")
        axes[1].set_title("Noise"); axes[1].set_xlabel("Depolarizing Probability"); axes[1].set_ylabel("QBER")
        # Panel 3
        df_skew = st.session_state["df_skew"]
        axes[2].plot(df_skew["Eve Basis Bias"], df_skew["QBER"], marker="o", color="purple")
        axes[2].axhline(qber_threshold, color="red", linestyle="--")
        axes[2].set_title("ProbabilisticSkew"); axes[2].set_xlabel("Eve Basis Bias"); axes[2].set_ylabel("QBER")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Run Step 6 first to precompute results.")
        
# ======================================================================
# Strategy-aware QBER Heatmap
# ======================================================================
st.markdown("---")
st.subheader("Strategy-aware QBER Heatmap")
strategy_choice = st.selectbox("Choose Eve strategy for heatmap", ["InterceptResend", "ProbabilisticSkew"])
if st.button("Generate Strategy Heatmap"):
    key = "df_heat_intercept" if strategy_choice=="InterceptResend" else "df_heat_skew"
    if key in st.session_state:
        df_heat = st.session_state[key]
        fig, ax = plt.subplots(figsize=(8, 6))

        # Heatmap
        sns.heatmap(df_heat, cmap="magma", ax=ax, cbar_kws={'label': 'QBER'})

        # Contour for threshold
        qber_grid = df_heat.values.astype(float)
        CS = ax.contour(qber_grid, levels=[qber_threshold], colors='cyan', linewidths=2,
                        extent=[0, qber_grid.shape[1], 0, qber_grid.shape[0]])
        ax.clabel(CS, inline=True, fmt={qber_threshold: f"Threshold={qber_threshold:.2f}"}, fontsize=10)

        # ✅ Add secure/insecure zone annotations
        ax.text(0.15, 0.05, "✅ Secure Zone", color="white", fontsize=12,
                transform=ax.transAxes, bbox=dict(facecolor="green", alpha=0.3))
        ax.text(0.65, 0.85, "❌ Abort Zone", color="black", fontsize=12,
                transform=ax.transAxes, bbox=dict(facecolor="red", alpha=0.3))

        # Labels
        ax.set_xlabel("Intercept Probability")
        ax.set_ylabel("Depolarizing Probability")
        ax.set_title(f"QBER Heatmap ({strategy_choice})")

        st.pyplot(fig)
        st.dataframe(df_heat)
    else:
        st.warning("Run Step 6 first to precompute results.")

st.info("Tip: Use the six workflow steps to simulate BB84. Then explore comparative plots, the tri-panel dashboard, and heatmaps for deeper insights.")