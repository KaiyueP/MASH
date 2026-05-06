import numpy as np

# ============================================================
# User settings for TC model
# ============================================================
eV_to_Hartree = 1.0 / 27.211386245988
JtoeV = 6.241509e18  # J/eV
t0 = 2.4188843265857e-17  # atomic unit of time in seconds
Eh = 4.3597447222071e-18  # hartree in joules
N_QD = 9
nstate_per_qd = 1
nmode_per_qd = 30
N_cavity = 3

# Cavity energies and QD-cavity coupling in eV
cavity_energies_eV = np.array([2.05, 2.10, 2.15], dtype=float)
g_qd_cavity_eV = 0.01

# Bath data for one QD; it will be copied to every QD block.
# The file is assumed to have the desired data in the last column.
# We reshape it into (n_modes, nstate_per_qd, nstate_per_qd) and convert units.
Vklq_SI_block = np.loadtxt("Vklq-diabatic.dat")[:, -1].reshape(-1, 40, 40)  # in sqrt(J)/s
Vklq_SI_block = Vklq_SI_block[6:6 + nmode_per_qd, :nstate_per_qd, :nstate_per_qd]
conv_factor_V = t0 / np.sqrt(Eh)
Vklq_AU_block = Vklq_SI_block * conv_factor_V
w_SI_block = np.loadtxt("w.dat")[:, 1][6:6 + nmode_per_qd] * 2 * np.pi * 1e12  # in s^-1
w_AU_block = w_SI_block * t0

# Read the single-QD excitation energies and replicate them per QD.
ExcEn = np.array([line.strip().split()[1] for line in open("exciton.dat", 'r')]).astype(float) # in Ha
ExcEn = ExcEn[:nstate_per_qd]

lmd = np.einsum("kij->ij", Vklq_SI_block**2 / w_SI_block[:, None, None]**2 / 2) * JtoeV  # in eV
np.savetxt("lmd_nm.dat", lmd, fmt="%.16f")

# ============================================================
# Build TC single-excitation Hamiltonian in atomic units
# State ordering: [QD0 states, QD1 states, ..., QD(N_QD-1) states, CAV states]
# ============================================================
ns = N_QD * nstate_per_qd + N_cavity
ham_sys_AU = np.zeros((ns, ns), dtype=float)

# QD diagonal energies: contiguous blocks by QD
qd_diag = np.diag(ExcEn)
for i_qd in range(N_QD):
    i0 = i_qd * nstate_per_qd
    i1 = i0 + nstate_per_qd
    ham_sys_AU[i0:i1, i0:i1] = qd_diag

# Cavity diagonal energies
if cavity_energies_eV.shape[0] != N_cavity:
	raise ValueError("cavity_energies_eV length must equal N_cavity")
# total QD electronic count (without cavity block)
qd_total = N_QD * nstate_per_qd
ham_sys_AU[qd_total:, qd_total:] = np.diag(cavity_energies_eV * eV_to_Hartree)

# QD-cavity couplings: couple each cavity mode to the first state of each QD block.
# This keeps the cavity phonon-free and preserves the block ordering.
for i_qd in range(N_QD):
	qd_state = i_qd * nstate_per_qd
	for i_c in range(N_cavity):
		ic = qd_total + i_c
		ham_sys_AU[qd_state, ic] = g_qd_cavity_eV * eV_to_Hartree
		ham_sys_AU[ic, qd_state] = ham_sys_AU[qd_state, ic].conjugate()


# ============================================================
# Build QD-local baths: each QD gets an identical bath block.
# Cavity states remain phonon-free.
# ============================================================
ns_qd = N_QD * nstate_per_qd
nw = N_QD * nmode_per_qd
w_AU = np.tile(w_AU_block, N_QD)
Vklq_AU = np.zeros((nw, ns, ns), dtype=float)
mode_owner = np.zeros(nw, dtype=np.int32)

for i_qd in range(N_QD):
	m0 = i_qd * nmode_per_qd
	m1 = m0 + nmode_per_qd
	s0 = i_qd * nstate_per_qd
	s1 = s0 + nstate_per_qd
	Vklq_AU[m0:m1, s0:s1, s0:s1] = Vklq_AU_block
	# 1-based QD id for local modes owned by this QD block.
	mode_owner[m0:m1] = i_qd + 1



np.savez(
	"tc_params_AU.npz",
	ham_sys_AU=ham_sys_AU,
	w_AU=w_AU,
	Vklq_AU=Vklq_AU,
	mode_owner=mode_owner,
	N_QD=np.int64(N_QD),
	N_cavity=np.int64(N_cavity),
	nmode_per_qd=np.int64(nmode_per_qd),
	nstate_per_qd=np.int64(nstate_per_qd),
	ns=np.int64(ns),
	nw=np.int64(nw)
)

print("Saved tc_params_AU.npz")
print(f"states = {ns} (QD states = {ns_qd}, N_QD={N_QD}, nstate_per_qd={nstate_per_qd}, N_cavity={N_cavity})")
