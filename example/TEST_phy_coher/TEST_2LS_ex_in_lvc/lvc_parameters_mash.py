import numpy as np

# ============================================================
# Minimal two-level system using the generic LVC backend
# ============================================================
# State ordering:
#   0, 1: two electronic states
#
# The bath is shared by both states because the same set of modes appears in
# the global LVC Hamiltonian. All linear vibronic couplings are set to zero
# manually, so the bath is dynamically decoupled from the 2LS.

ns = 2
nmode = 200

# Unit conversions
autoev = 27.2114079527
evtoau = 1.0 / autoev
cmm1 = 4.556335e-6

# A simple fixed 2LS Hamiltonian in atomic units.
# Energies are arbitrary for this test; the off-diagonal interaction is nonzero.
energy_gap_eV = 0.00
offdiag_coupling_eV = 0.01
ham_sys_AU = np.array(
	[
		[0.0, offdiag_coupling_eV * evtoau],
		[offdiag_coupling_eV * evtoau, energy_gap_eV * evtoau],
	],
	dtype=np.float64,
)

# One shared bath: all modes are available to both states through the same
# omega array. Frequencies are arbitrary and positive for this zero-coupling
# test.
w_min_cmm1 = 50.0
w_max_cmm1 = 2000.0
w_AU = np.linspace(w_min_cmm1, w_max_cmm1, nmode, dtype=np.float64) * cmm1

# Manually zero all state-mode couplings.
Vklq_AU = np.zeros((nmode, ns, ns), dtype=np.float64)

np.savez(
	"lvc_params_AU.npz",
	ham_sys_AU=ham_sys_AU,
	w_AU=w_AU,
	Vklq_AU=Vklq_AU,
)

print("Saved lvc_params_AU.npz")
print("2LS Hamiltonian [eV]:")
print(ham_sys_AU / evtoau)
print(f"states = {ns}, shared bath modes = {nmode}, max |Vklq_AU| = {np.max(np.abs(Vklq_AU)):.3e}")
