import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="ELOBO Outlier Detection Tool", layout="wide")
st.title("ELOBO Outlier Detection Tool")
st.markdown("Workflow: 1. Parameter Estimation → 2. Outlier Detection (ELOBO) → 3. Residual Analysis & Visualization")

# Left_Sidebar: Upload，Settings
with st.sidebar:
    st.header(" Upload Files ")
    uploaded_A = st.file_uploader("Matrix A (.txt) - shape: m×n (each row is an observation)", type="txt")
    uploaded_b = st.file_uploader("Vector b (.txt) - shape: m×1 (offset vector)", type="txt")
    uploaded_y = st.file_uploader("Vector y (.txt) - shape: m×1 (observed values)", type="txt")


# Session state
if 'show_elobo' not in st.session_state:
    st.session_state.show_elobo = False
if 'ls_done' not in st.session_state:
    st.session_state.ls_done = False


def detect_q_structure(Q):
    m = Q.shape[0]
    if np.allclose(Q, np.eye(m)):
        return 1, "Q is an identity matrix: observations not correlated"
    elif np.all(Q == np.diag(np.diag(Q))):
        return 2, "Q is diagonal: observations not correlated"
    elif np.count_nonzero(Q) / Q.size > 0.8:
        return 4, "Q is fully filled: possibly fully correlated"
    else:
        return 3, "Q has block-diagonal structure"


def visualize_q_blocks(Q, blocks):
    Q_copy = np.zeros_like(Q)
    for idx, block in enumerate(blocks):
        for i in block:
            for j in block:
                Q_copy[i, j] = idx + 1
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(Q_copy, cmap="Set3", cbar=False, linewidths=0.5, square=True, ax=ax)
    ax.set_title("Detected Block Structure in Q")
    st.pyplot(fig)


def extract_blocks_from_Q(Q, ignore_isolated=True):
    m = Q.shape[0]
    visited = np.zeros(m, dtype=bool)
    blocks = []

    adjacency = [[] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j and (Q[i, j] != 0 or Q[j, i] != 0):
                adjacency[i].append(j)

    def dfs(i, current_block):
        for j in adjacency[i]:
            if not visited[j]:
                visited[j] = True
                current_block.append(j)
                dfs(j, current_block)

    for i in range(m):
        if not visited[i]:
            visited[i] = True
            block = [i]
            dfs(i, block)
            if not ignore_isolated or len(block) > 1:
                blocks.append(sorted(block))

    return blocks


def analyze_internal_block_structure(Q, blocks):
    internal_analysis = []
    for block in blocks:
        subQ = Q[np.ix_(block, block)]
        if np.allclose(subQ, np.eye(len(block))):
            status = "Independent"
        elif np.all(subQ == np.diag(np.diag(subQ))):
            status = "Independent with varied weights"
        else:
            status = "Internally correlated"
        internal_analysis.append({"Block": block, "Status": status})
    return internal_analysis


def enrich_elobo_results_with_structure(df, structure_info):
    structure_map = {str(info["Block"]): info["Status"] for info in structure_info}
    df["Internal Structure"] = df["Block"].map(structure_map)
    return df


def run_least_squares(A, b, y, sigma2, Q):
    y_residual = y - b
    Cyy = sigma2 * Q
    try:
        Cyy_inv = np.linalg.inv(Cyy)
    except np.linalg.LinAlgError:
        Cyy_inv = np.linalg.pinv(Cyy)
    N = A.T @ Cyy_inv @ A
    u = A.T @ Cyy_inv @ y_residual
    try:
        x_hat = np.linalg.solve(N, u)
    except np.linalg.LinAlgError:
        x_hat = np.linalg.pinv(N) @ u
    y_hat = A @ x_hat + b
    residuals_full = y - y_hat
    dof = len(y) - len(x_hat)
    sigma2_hat = (residuals_full.T @ Cyy_inv @ residuals_full) / dof
    return x_hat, y_hat, residuals_full, sigma2_hat


def run_elobo_efficient(A, b, y, sigma2, blocks, threshold, Q):
    # Step 1: Full Least Squares
    y_residual = y - b
    Cyy = sigma2 * Q

    try:
        Cyy_inv = np.linalg.inv(Cyy)
    except np.linalg.LinAlgError:
        Cyy_inv = np.linalg.pinv(Cyy)

    N = A.T @ Cyy_inv @ A
    u = A.T @ Cyy_inv @ y_residual

    try:
        N_inv = np.linalg.inv(N)
    except np.linalg.LinAlgError:
        N_inv = np.linalg.pinv(N)


    x_full = N_inv @ u
    y_fit = A @ x_full + b
    residuals_full = y - y_fit

    sigma2_full = (residuals_full.T @ Cyy_inv @ residuals_full) / (len(y) - len(x_full))
    M = len(y)
    N_param = len(x_full)

    # Step 2: ELOBO for each block
    results = []

    for block in blocks:
        block_idx = np.array(block, dtype=int)

        if block_idx.size == 0:
            continue

        if np.any(block_idx >= Q.shape[0]):
            print(f"Skipping invalid block index: {block_idx} (Q size {Q.shape[0]})")
            continue

        A_block = A[block_idx, :]
        Q_block = Q[np.ix_(block_idx, block_idx)]
        v_block = residuals_full[block_idx]

        middle_matrix = Q_block - A_block @ N_inv @ A_block.T

        try:
            middle_inv = np.linalg.inv(middle_matrix)
        except np.linalg.LinAlgError:
            middle_inv = np.linalg.pinv(middle_matrix)

        omega_block = Q_block @ middle_inv @ v_block

        p_block = len(block_idx)
        sigma2_minus_block = ((M - N_param) * sigma2_full - v_block.T @ middle_inv @ v_block) / (M - N_param - p_block)

        omega_norm = np.linalg.norm(omega_block)
        diff_value = abs(omega_norm - np.linalg.norm(residuals_full))

        status = 'Outlier' if diff_value > threshold else 'OK'

        results.append({
            'Block': str(block),
            'Omega Norm': round(float(omega_norm), 6),
            'Difference': round(float(diff_value), 6),
            'Threshold': threshold,
            'Status': status
        })

    df = pd.DataFrame(results)
    return df


def show_residual_plot(y, y_new, error):
    fig, ax = plt.subplots()
    ax.plot(y, label='Original y', marker='o')
    ax.plot(y_new, label='Estimated ŷ', marker='x')
    ax.bar(np.arange(len(error)), error.flatten(), alpha=0.3, label='Residuals')
    ax.set_title('Residual Plot')
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)


# Main interface: Q structure selection
st.subheader("Step 0: Specify Q Structure")
q_structure = st.radio("Please select the structure of Q:", [
    "1. Q is diagonal (Standard I)",
    "2. Q is diagonal (Not I)",
    "3. Q is diagonal in blocks (squared)",
    "4. Q is completely filled"
])
Q = None
ready_for_step1 = False

if uploaded_A and uploaded_b and uploaded_y:
    A = np.loadtxt(uploaded_A)
    b = np.loadtxt(uploaded_b).reshape(-1, 1)
    y = np.loadtxt(uploaded_y).reshape(-1, 1)
    m = y.shape[0]

    if not (A.shape[0] == b.shape[0] == m):
        st.error(" Error: A, b, and y must have the same number of rows.")
    else:
        if q_structure == "1. Q is diagonal (Standard I)":
            st.success("Q is identity. Automatically constructed.")
            sigma2 = st.number_input("Sigma² (variance of each observation)", min_value=0.000001, value=1.0)
            Q = np.eye(m)
            threshold = None
            ready_for_step1 = True

        elif q_structure == "2. Q is diagonal (Not I)":
            st.info(f"Please upload a column of diagonal values (shape: {m}×1)")
            uploaded_q_col = st.file_uploader(f"Upload diagonal Q values (.txt) (shape: {m}×1)", type="txt")
            if uploaded_q_col:
                q_col = np.loadtxt(uploaded_q_col).reshape(-1, 1)
                st.info(f"Uploaded shape: {q_col.shape[0]}×{q_col.shape[1]}")
                if q_col.shape[0] != m:
                    st.error("Diagonal vector length must match A/b/y row count.")
                else:
                    Q = np.diag(q_col.flatten())
                    sigma2 = st.number_input("Sigma² (variance scaling)", min_value=0.000001, value=1.0)
                    threshold = st.number_input("Outlier Threshold", min_value=0.0, value=0.5,key="threshold_q2")
                    ready_for_step1 = True

        elif q_structure in ["3. Q is diagonal in blocks (squared)", "4. Q is completely filled"]:
            st.info(f"Please upload the full Q matrix (shape: {m}×{m})")
            uploaded_Q = st.file_uploader(f"Upload full Q matrix (.txt) (shape: {m}×{m})", type="txt")

            if uploaded_Q:
                try:
                    Q = np.loadtxt(uploaded_Q)
                    st.info(f" Uploaded shape: {Q.shape[0]}×{Q.shape[1]}")
                except Exception as e:
                    st.error(
                        f"Failed to load Q matrix. Make sure it's a plain text file with numeric values only. Error: {e}")
                    Q = None

                if Q.shape != (m, m):
                    st.error("Q must be a square matrix with the same number of rows as A/b/y.")
                else:
                    sigma2 = st.number_input("Sigma² (for scaling Q)", min_value=1e-8, value=4e-6, format="%.8f")
                    threshold = st.number_input("Outlier Threshold", min_value=0.0, value=0.5,key="threshold_q3")
                    ready_for_step1 = True

        if ready_for_step1:
            st.subheader("Step 1: Run Least Squares Estimation")
            if st.button("Run Estimation"):
                x, y_new, error, sigma2_est = run_least_squares(A, b, y, sigma2, Q)
                st.session_state.x = x
                st.session_state.y_new = y_new
                st.session_state.error = error
                st.session_state.sigma2_est = sigma2_est
                st.session_state.ls_done = True
                st.session_state.show_elobo = False

            if st.session_state.get("ls_done", False):
                with st.expander(" Estimation Results"):
                    st.write(f"x̂: {st.session_state.x.flatten()}")
                    st.write(f"ŷ: {st.session_state.y_new.flatten()}")
                    st.write(f"Residuals: {st.session_state.error.flatten()}")
                    st.write(f"Estimated σ̂²: {st.session_state.sigma2_est[0, 0]:.6f}")

                st.subheader(" Residual Plot")
                show_residual_plot(y, st.session_state.y_new, st.session_state.error)

                if not st.session_state.get("show_elobo", False):
                    if st.button("➡ Continue to Outlier Detection"):
                        st.session_state.show_elobo = True

            if st.session_state.get("show_elobo", False) and threshold is not None:
                st.subheader("Step 2: ELOBO Outlier Detection")
                if np.allclose(Q, np.diag(np.diag(Q))):
                    blocks = [[i] for i in range(len(y))]
                    st.write("Q is diagonal → using one observation per block.")
                else:
                    blocks = extract_blocks_from_Q(Q)
                    st.write(f"Detected Blocks: {blocks}")
                st.write(f"Detected Blocks: {blocks}")
                df = run_elobo_efficient(A, b, y, sigma2, blocks, threshold, Q)
                st.dataframe(df)
                st.download_button(" Download Detection Results", data=df.to_csv(index=False), file_name="elobo_results.csv")

                st.subheader(" Q Block Structure Visualization")
                visualize_q_blocks(Q, blocks)

