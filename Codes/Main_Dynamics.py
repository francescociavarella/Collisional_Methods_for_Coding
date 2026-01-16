# +++++++++++++++++++++++++++
#  Studing Collisonal Methods
# +++++++++++++++++++++++++++

import numpy as np
from scipy.linalg import expm
from qutip import *
import pickle
import os

# ===========================================================================================

# ==============
# Pauli Matrices
# ==============

sz = np.array([[1,0], [0,-1]]); sx = np.array([[0,1],[1,0]]); sy = np.array([[0,-1j],[1j,0]]) 

# ===========================================================================================

# ============
# Hamiltonians
# ============

def system_Hamiltonian(N_site, E, V_array):
    """
    Build up of the System's Hamiltonian for the complete basis (ground & excited states) or only excited states.

    Parameters: - E: Float, System's Site Energies (randomly generated)
                - V_array: Float, Hopping Potential
                - N_site : Int, Number of Sites

    Returns : System's Hamiltonian as Numpy array
    """
    # -----------------------------------------------------
    # Build symmetric matrix from upper triangular elements
    # -----------------------------------------------------
    V_matrix = np.zeros((N_site, N_site))
    idx = 0  # runs over V_array
    for i in range(N_site):
        for j in range(i+1, N_site):
            V_matrix[i, j] = V_array[idx]
            V_matrix[j, i] = V_array[idx]  # Symmetric
            idx += 1

    # --------------
    # Complete Basis 
    # --------------    

    H_sys = np.zeros((2**N_site, 2**N_site), dtype='complex')

    for i in range(N_site):
        H_i = (E[i]/2) * (tensor(identity(2**i), identity(2)-sigmaz(), identity(2**(N_site-i-1))))
        H_sys += H_i.full()

        for j in range(i+1, N_site):
               H_ij = V_matrix[i, j]/2 * (tensor(identity(2**i), sigmax(), identity(2**(j-i-1)), sigmax(), identity(2**(N_site-j-1))) + tensor(identity(2**i), sigmay(), identity(2**(j-i-1)), sigmay(), identity(2**(N_site-j-1))))
               H_sys += H_ij.full()

    return H_sys

# ===============================================================================================================================================

def interaction_Hamiltonian_N_ancillas(N_site, c_CM, method=None):  
    """
    Build up of the Hamiltonian of Interaction for the Collision System - Ancilla, 
    based on interaction of a different anclla with every different site

    Parameters: - N_site : int, Number of Sites
                - c_CM : list, Interaction Forces for the System - Ancilla intercation/collsion

    Method: - QJ : Quantum Jump Limit
            - Diff : Diffusive Limit

    Returns : Hamiltonian of Interaction as Qutip object
    """
    if method is None:
        method = INTERACTION_LIMIT

    dim_tot = 2**(2 * N_site)   # total Hilbert Space System + Ancilla
    H_int = np.zeros((dim_tot, dim_tot), dtype=complex)   #inizialization

    # Selection of the Ancilla's operator
    if method == 'QJ':
        anc_op = sigmax() # Interaction Z (sys) - X (anc) -> gives jumps/flip
    elif method == 'Diff':
        anc_op = sigmaz() # Interaction Z (sys) - Z (anc) -> gives dephasing
    else:
        raise ValueError("Method : 'QJ' or 'Diff'")

    for j in range(N_site):

        op_list = [identity(2) for _ in range(2 * N_site)]  #list of identity to be fill with the operator sigmaz & sigmax; 2N identity, N for the system and N fo the ancillas

        op_list[j] = sigmaz()      # Acts on the j site
        op_list[N_site + j] = anc_op  # Acts on the j ancilla, with index N + j

        H_term = (c_CM[j] * tensor(op_list)).full()  # tensor product between the element of the list

        H_int += H_term

    return H_int

# ================================================================================================================================================================================

def hamiltonian_N_ancillas(N_site, E, V_array, c_CM):
    """
    Generation of 3 Hamiltonians in Qutip format for the collision model with N ancillas:
                - H_system : system Hamiltonian
                - H_collision : interaction Hamiltonian with N ancillas
                - H_tot : complete Hamiltonian (system + collision)

    Parameters: - E: Float, System's Site Energies (randomly generated)
                - V_array: Float, Hopping Potential
                - N_site : int, Number of Sites
                - c_CM : list, Interaction Forces for the System - Ancilla intercation/collsion

    Returns : H_system, H_collision, H_tot 
    """

    H_collision = interaction_Hamiltonian_N_ancillas(N_site, c_CM)

    H_system = system_Hamiltonian(N_site, E, V_array)

    dim_anc = 2**N_site
    Id_ancillas = np.eye(dim_anc, dtype=complex)
    H_system_expanded = np.kron(H_system, Id_ancillas)  #expand H_sys in the total space

    H_tot = H_system_expanded + H_collision

    return H_system, H_collision, H_tot

#=========================================================================================================

def evolution_operator(H, dt, method='expm'):
    """
    Build up of the evolution operator U = exp(-i H dt) using Expm or analytic diagonalization.

    Parameters: - H : Np array, System Hamiltonian
                - dt : float, Timestep

    Method : - "expm"-> build up of the Matrix Exponential with expm
             - "diagonalization"->  build up of the propagater U as V @(exp(-i W dt))@ V_dag with W eigenvalues and V eigenvector of the Hamiltonian 

    Returns : Evolution Operator U, eigenvalues w and eigenvectors V
    """
    H = H.full() if hasattr(H, "full") else np.array(H)

    # -----------
    # Expm method
    # -----------

    if method == 'expm':
        U = expm(-1j * H * dt)
        return U

    # ---------------
    # Diagonalization
    # ---------------

    elif method == 'diagonalization':
        w, V = np.linalg.eigh(H)
        V_inv = V.conj().T

        U_diag = np.diag(np.exp(-1j * w * dt))
        U = V @ U_diag @ V_inv
        return U, U_diag, w, V

    else:
        raise ValueError("method : 'expm' or 'diagonalization'")

# ============================================================================================================================

# ===================
#  Lindblad Evolution
# ===================

def Liouvillian(H, gamma_j, L_j):
    """
    Build the Liouvillian superoperator.

    Parameters: - H : np array, Hamiltonian matrix
                - gamma_j : list, Decay rates
                - L_j : list, Jump Operators

    Returns: super_L  Liouvillian superoperator
    """    
    I = np.eye(H.shape[0])
    super_L = -1.j * (np.kron(I, H) - np.kron(H.T, I))

    for k in range(len(gamma_j)):
        super_L += gamma_j[k] * (np.kron(np.conj(L_j[k]), L_j[k]) - 0.5 * np.kron(I, np.conj(L_j[k]).T @ L_j[k]) - 0.5 * np.kron((np.conj(L_j[k]).T @ L_j[k]).T, I))

    return super_L

# ====================================================================================

def Lindblad_evo(rho, H, gamma_j, L_j, times, method="expm", vectorized=True):
    """
    Evolution of the density matrix with the Linblad Eq.

    Method: - "U" -> propagator = expm(super_L * dt) 
            - "diagonal" -> diagonalizzation of the super-op. 

    Vectorized: True/False to choose the output format

    Parameters: - H : np array, System Hamiltonian
                - rho : np array, Initial Density Matrix
                - gamma_j : list, List of Decay Rates
                - L_j : list, List of Jump Operators
                - times : array, Time array

    Returns : - if vectorized=True → array (N^2, Nt)
              - if vectorized=False → array (Nt, N_site, N_site)
              - eigenvectors w and eigenvalues V
    """
    # Convert to NumPy
    L_j = [L.full() if hasattr(L, "full") else np.array(L, dtype=complex) for L in L_j]
    H = H.full() if hasattr(H, "full") else np.array(H, dtype=complex)
    rho = rho.full() if hasattr(rho, "full") else np.array(rho, dtype=complex)

    rho_shape = H.shape[0]
    dt = times[1] - times[0]

    # Build up of the Liouvillian
    super_L = Liouvillian(H, gamma_j, L_j)

    # Vectorized intial state
    rho_vec = rho.reshape(rho_shape * rho_shape)

    # Result array inizialized
    rho_vec_list = np.zeros((rho_shape * rho_shape, len(times)), dtype=complex)
    rho_vec_list[:, 0] = rho_vec

    # -------------
    # Expm method
    # -------------

    # Time Propagator
    if method == "expm":
        super_U = expm(super_L * dt)

        # Evolution
        for i in range(1, len(times)):
            rho_vec_list[:, i] = super_U @ rho_vec_list[:, i - 1]

        # Output
        if vectorized:
            return rho_vec_list  # (rho_shape^2, Nt)
        else:
            return rho_vec_list.T.reshape(len(times), rho_shape, rho_shape)  # (Nt, rho_shape, rho_shape)

    # ------------------
    # Diagonal method
    # ------------------    

    elif method == "diagonal":  

        #Diagonalization of the Super-Op.
        W, V = np.linalg.eig(super_L)
        V_inv = np.linalg.inv(V)

        # Build up of the Diagonal Lindbladian
        U_diag = np.exp(W * dt)

        # Initial coefficients in the eigenvectors base
        coeff = V_inv @ rho_vec        
        coeff_list = np.zeros((len(W), len(times)), dtype=complex)
        coeff_list[:, 0] = coeff

        # Evolution of the coefficients
        for i in range(1, len(times)):
            coeff_list[:, i] = U_diag * coeff_list[:, i - 1]  

        # Reconversion in the original base
        rho_vec_list = V @ coeff_list

        # Output with Eigenenergies & Eigenvectors (as NumPy array)
        if vectorized:
            return rho_vec_list, V, W  
        else:
            return rho_vec_list.T.reshape(len(times), rho_shape, rho_shape), V, W 

    else:
        raise ValueError("Set 'expm', 'diagonal'.")

# =================================================================================================================

# =========================
# Isolated System Evolution 
# =========================

def compute_trajectory_wf_isolated(N_site, times, projectors, psi_sys_initial, U_site):
    """
    Compute quantum trajectory evolution of the wave function with only H_Exc = Energy of the Site and Hopping Potential V.
    Calculated for only 1 trajecotry (always the same evolution)

    Parameters: - N_site : int, Number of Sites
                - steps : int, Number of Time Steps
                - times : array, Time array
                - projectors : Np array, Projection Operators on |10> & |01>
                - psi_sys_initial : Qobj, Initial Wave Function
                - U_site : Qobj, Time Evolution Operator

    Returns: -  pop_traj_isolated : np array, Population for each trajectory (N_site x steps x N_traj)
    """

    # check and conversion in Numpy object
    U_site = U_site.full() if hasattr(U_site, 'full') else U_site
    psi_sys_initial = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else psi_sys_initial.copy()

    # Isolated system
    pop_traj_isolated = np.zeros((N_site, len(times)))

    # Inizialization
    for site in range(N_site):
        pop_traj_isolated[site, 0] = np.real(np.vdot(psi_sys_initial, projectors[site] @ psi_sys_initial))

    psi = psi_sys_initial.copy()   #reset initial wf |01>

    # Evolution in Time (starting after 1st timestep)
    for step in range(1, len(times)):

        psi = U_site @ psi   # Evolution of wf with H_site

        for site in range(N_site):
            pop_traj_isolated[site, step] = np.real(np.vdot(psi, projectors[site] @ psi))

    return pop_traj_isolated

# ===================================================================================================================================================

# =============================
#  Collisional Method functions
# =============================

# ===========================
#  Trace on Ancilla Evolution
# ===========================

def compute_trace_ancilla(rho_sys_initial, U_diag, V, times, projectors, N_site, method=None):
    """
    Compute time evolution with Trace over Ancilla and Reset, using Density Matrix formalism (equal to Infinite Trajectories)

    Parameters: - rho_sys_initial : Np array, Initial Density Matrix
                - U_diag: Np array, Time Evolution Operator for the complete system (S + A) in diagonal form
                - U_diag_dag: Np array, Adjoint operator of U_diag
                - V_diag: Np array, Hopping Potential
                - V_diag_dag: Np array, Adjoint operator of V_diag
                - times : array, Time array
                - N_site : int, Number of Sites                
                - projectors: Np array, Projection Operators on |10> & |01>

    Method: - QJ : Quantum Jump Limit, uses |0><0|
            - Diff : Diffusive Limit, uses I/2

    Returns: - pops_complete : np array (N_site x len(times)),  Population evolution for each site
    """
    if method is None:
        method = INTERACTION_LIMIT

    # ------------------------
    # Ancilla's Density Matrix
    # ------------------------

    if method == 'QJ':

        rho_anc_single = ket2dm(basis(2,0)) # Pure state |0><0|: [[1, 0], [0, 0]]

    elif method == 'Diff':

        rho_anc_single = (qeye(2) / 2) # Completely Mixed state I/2: [[0.5, 0], [0, 0.5]]
    else:
        raise ValueError("Method must be 'QJ' or 'Diff'")

    rho_anc = (tensor([rho_anc_single for _ in range(N_site)])).full() #for N ancilla

    # ----------------
    # Convert to numpy
    # ----------------
    rho_sys = rho_sys_initial.full() if hasattr(rho_sys_initial, 'full') else rho_sys_initial.copy()

    projectors = np.array([P.full() if hasattr(P, 'full') else P for P in projectors], dtype=complex)

    # ------------------
    # Evolution Operator
    # ------------------
    V = V.full() if hasattr(V, 'full') else V
    U_diag = U_diag.full() if hasattr(U_diag, 'full') else U_diag
    U_step = V @ U_diag @ V.conj().T;
    U_step_dag = U_step.conj().T

    # ----------
    # Dimensions
    # ----------
    dim_sys = rho_sys.shape[0]
    dim_anc = rho_anc.shape[0]

    # --------------
    # Storing result
    # --------------
    # Array to store the results of the evolution for only the excited site
    pops_complete = np.zeros((N_site, len(times)), dtype=float) 

    # -------------
    # initial state
    # -------------
    for site in range(N_site):
        pops_complete[site, 0] = np.real(np.trace(projectors[site] @ rho_sys))

    # --------------
    # Time Evolution
    # --------------
    for t in range (1, len(times)):

        # 1 : expansion in the system-ancilla space by tensor product
        rho_tot = np.kron(rho_sys, rho_anc)   # I always use ancillas in their initial state! already resetted 

        # 2 : Unitary evolution of the total rho
        rho_tot = U_step @ rho_tot @ U_step_dag

        # 3 : Partial Trace on the ancilla's degree of freedom
        rho_tot_reshaped = rho_tot.reshape(dim_sys, dim_anc, dim_sys, dim_anc) # 4 blocks 4x4 (tensor)
        rho_sys = np.trace(rho_tot_reshaped, axis1=1, axis2=3) # contraction over elements with same ancilla's index

        # 4 : Store the result of the population
        for site in range(N_site):
            pops_complete[site, t] = np.real(np.trace(projectors[site] @ rho_sys))

    return pops_complete

#================================================================================================================

# =============
# Bloch Sphere
# =============

def compute_Bloch_Sphere(psi):
    """
    Function to extract the expectation value of the Bloch's Sphere components <sigmax>, <sigmay>, <sigmaz> associated to the 2x2 space of only excited states,
    with base |10> (exc. on site 1, -z) & |01> (exc. on site 2, +z) 

    Parameters: -psi : nparray, wf at time t of the complete systems (wf site1 otimes wf site2)

    Returns: - r_x_step, float expectation value of x component <sigmax>
             - r_y_step, float expectation value of y component <sigmay>
             - r_z_step, float expectation value of z component <sigmaz>
    """
    # Flatten wave function if needed
    if psi.ndim > 1:
        psi = psi.flatten()

    # wf element
    c_01 = psi[1] ; c_01_conj = np.conj(c_01) # site 2 
    c_10 = psi[2] ; c_10_conj = np.conj(c_10) # site 1

    # Blochs components
    r_x_step = 2 * np.real(c_10 * c_01_conj)

    r_y_step = -2 * np.imag(c_10 * c_01_conj)

    r_z_step = np.abs(c_01)**2 - np.abs(c_10)**2

    return r_x_step, r_y_step, r_z_step

# ===========================================================================================================================================================================

# =====================
# Stochastic trajectory 
# =====================

def compute_trajectory_wf(dt, c_CM, N_traj, N_site, times, projectors, psi_sys_initial, U_site, method=None, n_samples=5):
    """
    Compute quantum trajectory evolution of the wave function with optimized memory usage.
    Saves only n_samples complete trajectories, accumulates all others for averaging.

    Parameters: - dt : float, Time Step
                - c_CM : array, Collisional model Coefficients
                - N_traj : int, Number of Trajectories
                - N_site : int, Number of Sites
                - times : array, Time array
                - projectors : Np array, Projection Operators on |10> & |01>
                - psi_sys_initial : Np array, Initial Wave Function
                - U_site : Np array, Time Evolution Operator
                - method : str, 'QJ' or 'Diff'
                - n_samples : int, Number of complete trajectories to save (default=3)

    Returns:  - pop_traj_samples : np array (N_site, len(times), n_samples), Complete sample trajectories
              - average_pop_traj : np array (N_site, len(times)), Average population over ALL trajectories
              - count : np array (N_traj,), Number of collisions per trajectory
              - avg_count : float, Average collision count
              - r_x_samples, r_y_samples, r_z_samples : np arrays (len(times), n_samples), Bloch samples
              - avg_r_x, avg_r_y, avg_r_z : np arrays (len(times),), Average Bloch coordinates over ALL trajectories
    """   
    if method is None:
        method = INTERACTION_LIMIT

    # ------------------------------------
    # check and conversion in Numpy object
    # ------------------------------------
    U_site = U_site.full() if hasattr(U_site, 'full') else U_site
    psi_sys_initial = psi_sys_initial.full() if hasattr(psi_sys_initial, 'full') else psi_sys_initial.copy()

    # --------------
    # Storing result
    # --------------
    n_times = len(times)

    # Number of effective collisions
    count = np.zeros(N_traj, dtype=np.int64)  

    # Arrays to store only n_samples complete trajectories
    pop_traj_samples = np.zeros((N_site, n_times, n_samples))
    r_x_samples = np.zeros((n_times, n_samples), dtype=np.float64)
    r_y_samples = np.zeros((n_times, n_samples), dtype=np.float64)
    r_z_samples = np.zeros((n_times, n_samples), dtype=np.float64)

    # Arrays to accumulate the sum of all trajectories (for averaging)
    pop_sum = np.zeros((N_site, n_times), dtype=np.float64)
    r_x_sum = np.zeros(n_times, dtype=np.float64)
    r_y_sum = np.zeros(n_times, dtype=np.float64)
    r_z_sum = np.zeros(n_times, dtype=np.float64)

    # -------------------------------------------------
    # Costruction of the sigmaz operator for every site
    # -------------------------------------------------
    Sz_ops = []
    for idx in range(N_site):
        ops = [qeye(2) for _ in range(N_site)]
        ops[idx] = sigmaz()  # sigmaz only on the index site
        Sz_op = tensor(ops).full()
        Sz_ops.append(Sz_op)

    Sz_ops = np.array(Sz_ops)

    # ------------
    # Probability
    # ------------
    if method == 'QJ':
        # Quantum Jump: probability for each site
        jump_probabilities = [np.sin(c * dt)**2 for c in c_CM]

    elif method == 'Diff':
        # Diffusive: fixed probability 
        jump_probability = 0.5

        # Precomputation
        cos_vals = np.cos(c_CM * dt)
        sin_vals = np.sin(c_CM * dt)

    # ------------------------------------
    # Evolution for different Trajectories
    # ------------------------------------
    for traj in range(N_traj):
        # Determine if this trajectory should be saved completely
        save_complete = (traj < n_samples)

        # 1. reset initial wf |01>
        psi = psi_sys_initial.copy()   

        # --------------
        # Initial state
        # --------------
        for site in range(N_site):
            pop_val = np.real(np.vdot(psi, projectors[site] @ psi))

            # Save in sample array if this is one of the first n_samples trajectories
            if save_complete:
                pop_traj_samples[site, 0, traj] = pop_val

            # Always accumulate in sum 
            pop_sum[site, 0] += pop_val

        # Initial Bloch's value
        rx, ry, rz = compute_Bloch_Sphere(psi)

        if save_complete:
            r_x_samples[0, traj] = rx
            r_y_samples[0, traj] = ry
            r_z_samples[0, traj] = rz

        # Always accumulate
        r_x_sum[0] += rx
        r_y_sum[0] += ry
        r_z_sum[0] += rz

        # 2. Evolution in Time (starting after 1st timestep)
        for step in range(1, n_times):

            # 3. Evolution of wf with H_exc
            psi = U_site @ psi   

            # 4. Defining the condition of the Monte Carlo - Jump for every site
            # ------------
            # Quantum Jump
            # ------------
            if method == 'QJ':

                for site_index in range(N_site):
                    rn_site = np.random.rand() # Random number between 0 & 1

                    if rn_site < jump_probabilities[site_index]:
                        psi = Sz_ops[site_index] @ psi   # apply sigmaz 

                        count[traj] += 1  # count number of collision
            # ---------
            # Diffusive
            # ---------
            elif method == 'Diff':
                for site_index in range(N_site):
                    Sz_psi = Sz_ops[site_index] @ psi

                    rn_site = np.random.rand() # Random number between 0 & 1

                    if rn_site < jump_probability:
                        psi = cos_vals[site_index] * psi - 1j * sin_vals[site_index] * Sz_psi             
                    else:
                        psi = cos_vals[site_index] * psi + 1j * sin_vals[site_index] * Sz_psi
                        count[traj] += 1  # count number of collision

            # 5. Normalization
            psi = psi / np.linalg.norm(psi)

            # 6. Expectation Value - Populations
            for site in range(N_site):
                pop_val = np.real(np.vdot(psi, projectors[site] @ psi))

                # Save in sample array 
                if save_complete:
                    pop_traj_samples[site, step, traj] = pop_val

                # Always accumulate
                pop_sum[site, step] += pop_val

            # Bloch's elements
            rx, ry, rz = compute_Bloch_Sphere(psi)

            if save_complete:
                r_x_samples[step, traj] = rx
                r_y_samples[step, traj] = ry
                r_z_samples[step, traj] = rz

            # Always accumulate
            r_x_sum[step] += rx
            r_y_sum[step] += ry
            r_z_sum[step] += rz

    # 7. Calculate averages from accumulated sums
    average_pop_traj = pop_sum / N_traj
    avg_r_x = r_x_sum / N_traj
    avg_r_y = r_y_sum / N_traj
    avg_r_z = r_z_sum / N_traj
    avg_count = np.mean(count)

    return pop_traj_samples, average_pop_traj, count, avg_count, r_x_samples, r_y_samples, r_z_samples, avg_r_x, avg_r_y, avg_r_z

# =================================================================================================================

# ==========
# Parameters
# ==========

# -------------------
# System's Parameters
# -------------------
N_site = 2            # Number of sites
V_array = [1.0]    # Hopping Potential : V12, V13, ... V1N_site, V23, ..., V2N_site, V34...V3N_site
E = 1.5 + np.random.randn(N_site)*0.1     #random inizialization of the system energies

# -------------------------
# Time Evolution Parameters
# -------------------------
dt_list = [0.1, 0.01, 0.001 ]   # Time step
tf = 50.0    # Final Time
steps_list = [ int(tf / dt_list[i]) for i in range (len(dt_list)) ]
times_list = [ np.linspace(0, tf, int(steps_list[i])) for i in range(len(dt_list))]

N_traj_list = [100, 1000, 10000]

# -------------------
# Dephasing Parameter
# -------------------
g_deph = 0.1  # Gamma rate

# -------------
#Lindblad Rates
# -------------
gamma_j = [g_deph, g_deph]

# ----------------------------------------------------------
# Scaling for the collsional algorithm c = sqrt(gamma / 4dt)
# ----------------------------------------------------------
c_CM_list = np.array([[np.sqrt(g_deph / (4 * dt_list[j])) for j in range(len(dt_list))] for _ in range(N_site)])  # same Coupling for the 2 sites

# ========================================
# Initial wave function and density matrix
# ========================================

# ------
# System
# ------
psi_sys_initial = tensor(basis(2, 0), basis(2, 1)) # I set the population only in site 2
rho_sys_initial = (ket2dm(psi_sys_initial)).full()

# ----------------------------
# Projectors on System's sites
# ----------------------------
P0 = (np.eye(2, dtype=complex) + sz) / 2 # projector on |0>
P1 = (np.eye(2, dtype=complex) - sz) / 2 # projector su |1>

P_00 = np.kron(P0, P0) # |00><00|
P_01 = np.kron(P0, P1) # |01><01|
P_10 = np.kron(P1, P0) # |10><10|
P_11 = np.kron(P1, P1) # |11><11|

projectors = np.array([P_10, P_01], dtype=complex) # for only excited states

# ----------------------
# Lindblad Jump Operator
# ----------------------
L_1 = P_10  # projector on |10><10|
L_2 = P_01  # projector on |01><01|
L_j = [L_1, L_2]

# ==========================
# QJ or Diff Limit Selection
# ==========================
                                                 # +++++++++++++++++++++ 
INTERACTION_LIMIT = 'QJ'  # 'QJ' or 'Diff'       # + Change limit here + 
                                                 # +++++++++++++++++++++ 
# ===========
# Calculation
# ===========

# ----------------------
# Dictionary for results
# ----------------------
results = {}

# ------------
# Loop over dt
# ------------
for dt_idx, dt in enumerate(dt_list):

    # Initialize dictionary for this dt
    results[dt] = {}

    # Extract parameters for this dt
    times = times_list[dt_idx]
    steps = steps_list[dt_idx]
    c_CM = c_CM_list[:, dt_idx]  # ! because c_CM depends on dt

    # =====================================
    # Recalculate Hamiltonian and Operators
    # =====================================

    # --------------------------------------------------
    # Hamiltonian & U for Density Matrix (with ancillas)
    # --------------------------------------------------
    H_site, H_coll, H_tot = hamiltonian_N_ancillas(N_site, E, V_array, c_CM)

    U_tot, U_diag, w, V = evolution_operator(H_tot, dt, method='diagonalization')
    U_diag_dag = U_diag.conj().T; V_dag = V.conj().T 

    # -----------------------------------------------
    # Hamiltonian & U for Wave Function (system only)
    # -----------------------------------------------
    H_system = system_Hamiltonian(N_site, E, V_array)

    U_site, U_diag_site, w_site, V_site = evolution_operator(H_system, dt, method='diagonalization')

    # =========
    # Lindblad
    # =========

    rho_list_lindblad, V_lindblad, W_lindblad = Lindblad_evo(rho_sys_initial, H_system, gamma_j, L_j, times, method="diagonal", vectorized=False)

    # ==============
    # Trace Ancilla
    # ==============

    pops_trace = compute_trace_ancilla(rho_sys_initial, U_diag, V, times, projectors, N_site)

    # ========================================
    # Trajectory Isolated (without collisions)
    # ========================================

    pop_traj_isolated = compute_trajectory_wf_isolated(N_site, times, projectors, psi_sys_initial, U_site)

    # ================
    # Loop over N_traj
    # ================
    for N_traj in N_traj_list:

        # ===================================
        # Trajectory for WF (with collisions)
        # ===================================

        pop_traj_samples, avg_pop_traj, count, avg_count, r_x_samples, r_y_samples, r_z_samples, avg_r_x, avg_r_y, avg_r_z = compute_trajectory_wf(dt, c_CM, N_traj, N_site,
                                                                                                        times, projectors, psi_sys_initial, U_site, method=None, n_samples=5)
        # ==================================
        # Save the Results in the Dictionary
        # ==================================

        results[dt][N_traj] = {
            'parameters': {
                'dt': dt,
                'N_traj': N_traj,
                'times': times,
                'steps': steps,
                'c_CM': c_CM.copy()  
            },

            # Trace Ancilla 
            'anc_trace': pops_trace,

             # Trajectory Isolated
            'trajectory_isolated': {
                'pop_traj': pop_traj_isolated,
            },

            # Lindblad 
            'lindblad': {
                'rho_list': rho_list_lindblad,
                'V': V_lindblad,
                'W': W_lindblad
            },

            # Trajectory WF
            'trajectory_wf': {
                'pop_traj_samples': pop_traj_samples,  # (N_site x len(times) x 3) 
                'average_pop': avg_pop_traj,            # (N_site x len(times))
                'count': count,                 
                'avg_count': avg_count,
                'bloch': {
                    'avg_x': avg_r_x,
                    'avg_y': avg_r_y,
                    'avg_z': avg_r_z,
                    'samples_x': r_x_samples,  # (len(times), 3)
                    'samples_y': r_y_samples,
                    'samples_z': r_z_samples
                }
            }        
        }
# ======================================================================================================================

# ==============
# Saving Results
# ==============

# ----------------------
# Creating new directory
# ----------------------
results_dir = "../Results/Data"

os.makedirs(results_dir, exist_ok=True)

# -----------------
# Saving Dictionary
# -----------------
filename = os.path.join(results_dir, f"results_{INTERACTION_LIMIT}.pkl") 

with open(filename, 'wb') as f:
    pickle.dump(results, f)

