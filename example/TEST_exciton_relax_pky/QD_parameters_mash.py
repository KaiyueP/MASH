import numpy as np

# ============================================================
# User settings for one-QD exciton relaxation using LVC backend
# ============================================================
nstate = 9
nmode = 200
QD_name = "3.0CdSe_2ML"
QD_data_dir = "/pscratch/sd/k/kaiyuep/QD_data"

# Fundamental constants and unit conversions
kb=1.380649e-23
c=3e8 #m/s
nmtoau=18.8972598858 #1nm in Bohr
autoev= 27.2114079527 # 1 hartree in ev
evtoau=1/autoev
autoj = 4.3597447222071e-18 # 1 Hartree in J
jtoev=6.2415091e18 #1j in eV
jtoau = jtoev * evtoau # 1 J in au
evtoj=1.602176565e-19 #1eV in J
hbar= 1.054571817e-34 #J⋅s
hbar_eV=6.582119569e-16 #eV⋅s
C_light = 3e8  # m/s
C_au = C_light/2.18769126364e6  # AU
autos = 2.4188843265857e-17
conv_factor_V = autos / np.sqrt(autoj)


def load_sorted_modes(qd_name):
	"""Load QD modes and rank them by total linear vibronic coupling strength."""
	qd_path = "%s/%s" % (QD_data_dir, qd_name)
	w_SI = np.loadtxt("%s/w.dat" % qd_path)[:, 1][6:] * 2 * np.pi * 1e12
	Vklq_SI = np.load("%s/Vklq.npy" % qd_path)[6:, :nstate, :nstate]

	mode_strength = np.einsum("kij->k", Vklq_SI**2 / w_SI[:, None, None]**2)
	sort_idx = np.argsort(mode_strength)[::-1]
	return w_SI[sort_idx], Vklq_SI[sort_idx], mode_strength[sort_idx]


# Read QD exciton energies. The exciton.dat values are already in Hartree.
ExcEn = np.array(
	[line.strip().split()[1] for line in open("%s/%s/exciton.dat" % (QD_data_dir, QD_name), "r")]
).astype(float)
ExcEn = ExcEn[:nstate]
ExcEn -= ExcEn[0]
ham_sys_AU = np.diag(ExcEn)

# Read and select the strongest local phonon modes for this one QD.
w_SI, Vklq_SI, strength = load_sorted_modes(QD_name)
w_SI = w_SI[:nmode]
Vklq_SI = Vklq_SI[:nmode, :nstate, :nstate]
w_AU = w_SI * autos
Vklq_AU = Vklq_SI * conv_factor_V

# Save the reorganization energies in eV for inspection.
lmd = np.einsum("kij->ij", Vklq_SI**2 / w_SI[:, None, None]**2 / 2) * jtoev
np.savetxt("lmd_nm.dat", lmd, fmt="%.16f")

np.savez(
	"lvc_params_AU.npz",
	ham_sys_AU=ham_sys_AU,
	w_AU=w_AU,
	Vklq_AU=Vklq_AU
)

print("Saved lvc_params_AU.npz")
print("states = %d, modes = %d, QD = %s" % (nstate, nmode, QD_name))
