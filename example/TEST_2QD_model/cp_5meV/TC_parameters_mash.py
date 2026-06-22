import numpy as np

# ============================================================
# User settings for two-QD / one-cavity TC model
# ============================================================
# system setup
complex_mode = False
nstate_per_qd = 4
NdimD = nstate_per_qd
NdimA = nstate_per_qd
nqd = 2
nmode_per_qd = 200
N_cavity = 1
QD_1="3.0CdSe_2ML" # QD coupled to local phonon bath 1
QD_2="3.0CdSe_2ML" # QD coupled to local phonon bath 2
N_QD = nqd
g_s_ref_eV=5e-3 #eV to au #coupling strength
distance = 6 #nm
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
E_c_eV=2.06 #eV

if N_QD != 2:
	raise ValueError("This parameter file builds exactly two QDs. Set nqd/N_QD to 2.")
if N_cavity != 1:
	raise ValueError("This parameter file builds exactly one cavity mode. Set N_cavity to 1.")

def load_sorted_mode_block(qd_name):
	"""Load and rank vibrational modes by coupling strength.

	The ranking score is the total mode coupling weight,
	sum(V_nm^2 / w^2), computed after skipping the first `skip` rows
	from the raw files.
	sort using first 10 states
	"""
	# in SI
	w_SI = np.loadtxt("/pscratch/sd/k/kaiyuep/QD_data/%s/w.dat" % qd_name)[:, 1][6:] * 2 * np.pi * 1e12
	Vklq_SI = np.load("/pscratch/sd/k/kaiyuep/QD_data/%s/Vklq.npy" % qd_name)[6:, :10, :10] 

	mode_strength = np.einsum("kij->k", Vklq_SI**2 / w_SI[:, None, None]**2)
	sort_idx = np.argsort(mode_strength)[::-1]
	select_idx = sort_idx

	w_SI_block = w_SI[select_idx]
	Vklq_SI_block = Vklq_SI[select_idx]
	return w_SI_block, Vklq_SI_block, mode_strength[select_idx]

def rotation_mat_func(v_initial, v_final):
    # Normalize the input vectors
    v_initial = v_initial / np.linalg.norm(v_initial)
    v_final = v_final / np.linalg.norm(v_final)
    # Calculate the axis of rotation using cross product
    axis = np.cross(v_initial, v_final)
    axis /= np.linalg.norm(axis) if not np.allclose(axis, 0) else 1.0
    # Calculate the angle between the vectors
    cos_theta = np.dot(v_initial, v_final)
    angle = np.arccos(cos_theta)
    # Rodrigues' rotation formula to compute the rotation matrix
    ux, uy, uz = axis
    rotation_matrix = np.array([
        [np.cos(angle) + ux**2 * (1 - np.cos(angle)), ux * uy * (1 - np.cos(angle)) - uz * np.sin(angle), ux * uz * (1 - np.cos(angle)) + uy * np.sin(angle)],
        [uy * ux * (1 - np.cos(angle)) + uz * np.sin(angle), np.cos(angle) + uy**2 * (1 - np.cos(angle)), uy * uz * (1 - np.cos(angle)) - ux * np.sin(angle)],
        [uz * ux * (1 - np.cos(angle)) - uy * np.sin(angle), uz * uy * (1 - np.cos(angle)) + ux * np.sin(angle), np.cos(angle) + uz**2 * (1 - np.cos(angle))]
    ])
    return rotation_matrix

# read date from files
#all in AU
ExcEn1 = np.array([line.strip().split()[1] for line in open("/pscratch/sd/k/kaiyuep/QD_data/%s/exciton.dat" % QD_1, 'r')]).astype(float) # in Ha
ExcEn1 = ExcEn1[:nstate_per_qd]
filename='/pscratch/sd/k/kaiyuep/QD_data/%s/intElecDipoleMoments.dat'%QD_1
dipole_data1 = np.loadtxt(filename,dtype='float',comments='#')
DipoMom1= dipole_data1[:NdimD,3:] # transition dipole vectors in au
mu1=np.sqrt(dipole_data1[:NdimD,2]) # transition dipole magnitudes in au
w_SI_block1, Vklq_SI_block1, strength1 = load_sorted_mode_block(QD_1)
w_SI_block1=w_SI_block1[:nmode_per_qd]
Vklq_SI_block1 = Vklq_SI_block1[:nmode_per_qd, :nstate_per_qd, :nstate_per_qd]
w_AU_block1 = w_SI_block1 * autos #convert to au
Vklq_AU_block1 = Vklq_SI_block1 * conv_factor_V

#all in AU
ExcEn2 = np.array([line.strip().split()[1] for line in open("/pscratch/sd/k/kaiyuep/QD_data/%s/exciton.dat" % QD_2, 'r')]).astype(float) # in Ha
ExcEn2 = ExcEn2[:nstate_per_qd]
filename='/pscratch/sd/k/kaiyuep/QD_data/%s/intElecDipoleMoments.dat'%QD_2
dipole_data2 = np.loadtxt(filename,dtype='float',comments='#')
DipoMom2= dipole_data2[:NdimA,3:] # transition dipole vectors in au
mu2=np.sqrt(dipole_data2[:NdimA,2]) # transition dipole magnitudes in au
w_SI_block2, Vklq_SI_block2, strength2 = load_sorted_mode_block(QD_2)
w_SI_block2=w_SI_block2[:nmode_per_qd]
Vklq_SI_block2 = Vklq_SI_block2[:nmode_per_qd, :nstate_per_qd, :nstate_per_qd]
w_AU_block2 = w_SI_block2 * autos #convert to au
Vklq_AU_block2 = Vklq_SI_block2 * conv_factor_V


E_cavity=E_c_eV*evtoau #evtoau
g_s_ref=g_s_ref_eV*evtoau #eV to au


#reorg in eV
lmd1 = np.einsum("kij->ij", Vklq_SI_block1**2 / w_SI_block1[:, None, None]**2 / 2) *jtoev  # in eV
np.savetxt("lmd1_nm.dat", lmd1, fmt="%.16f")
lmd2 = np.einsum("kij->ij", Vklq_SI_block2**2 / w_SI_block2[:, None, None]**2 / 2) *jtoev  # in eV
np.savetxt("lmd2_nm.dat", lmd2, fmt="%.16f")

# ============================================================
# Build TC single-excitation Hamiltonian in atomic units
# State ordering: [QD1 states, QD2 states, cavity]
# ============================================================
ns_qd = N_QD * nstate_per_qd #total QD state
ns = N_cavity + N_QD * nstate_per_qd
if complex_mode:
    print("Using complex mode, complex ham_sys_AU, but error may rise due to MASH real-valued expectation of coupling. ")
else:
	print("Using real mode, real ham_sys_AU")

# dipole dipole interaction
rotationD= rotation_mat_func(DipoMom1[0], mu1[0]*np.array([0, 0, 1]))
orien_DipoMomD=(rotationD@(DipoMom1.T)).T  ####each row is a rotated momentum vector
rotationA= rotation_mat_func(DipoMom2[0], mu2[0]*np.array([0, 0, 1]))
orien_DipoMomA=(rotationA@(DipoMom2.T)).T  ####each row is a rotated momentum vector
cs_const = g_s_ref / mu1[0]
# Construct energy vector with QD states first, then cavity, all in atomic units
Egn = np.concatenate((ExcEn1, ExcEn2, [E_cavity]))

# Initialize system Hamiltonian (atomic units)
ham_sys_AU = np.diag(Egn).astype(complex if complex_mode else float)

# Cavity-QD couplings are in atomic units because cs_const is scaled from g_s_ref.
cavity_idx = ns_qd
qd1_start = 0
qd1_end = qd1_start + nstate_per_qd
qd2_start = qd1_end
qd2_end = qd2_start + nstate_per_qd
qd1_slice = slice(qd1_start, qd1_end)
qd2_slice = slice(qd2_start, qd2_end)
ham_sys_AU[cavity_idx, qd1_slice] = ham_sys_AU[qd1_slice, cavity_idx] = np.abs(cs_const * orien_DipoMomD[:nstate_per_qd, 2])
ham_sys_AU[cavity_idx, qd2_slice] = ham_sys_AU[qd2_slice, cavity_idx] = np.abs(cs_const * orien_DipoMomA[:nstate_per_qd, 2])

# Dipole-dipole QD-QD coupling (distance in nm -> convert to au length with nmtoau)
for i in range(nstate_per_qd):
	for j in range(nstate_per_qd):
		kappa = abs(orien_DipoMomD[i] @ orien_DipoMomA[j] - 3 * orien_DipoMomD[i, 2] * orien_DipoMomA[j, 2])
		# local field factor included; result in atomic units
		ham_sys_AU[qd1_start + i, qd2_start + j] = ham_sys_AU[qd2_start + j, qd1_start + i] = kappa / (distance * nmtoau) ** 3 * (9.0 / 49.0)

##############DSE################
# Dipole self-energy contributions in atomic units.
dseD = cs_const * orien_DipoMomD[:nstate_per_qd, 2]
dseA = cs_const * orien_DipoMomA[:nstate_per_qd, 2]
ham_sys_AU[cavity_idx, cavity_idx] = ham_sys_AU[cavity_idx, cavity_idx] + np.sum(dseD ** 2 / E_cavity) + np.sum(dseA ** 2 / E_cavity)
for i in range(nstate_per_qd):
	for j in range(nstate_per_qd):
		ham_sys_AU[qd1_start + i, qd1_start + j] = ham_sys_AU[qd1_start + i, qd1_start + j] + (dseD[i] * dseD[j]) / E_cavity
for i in range(nstate_per_qd):
	for j in range(nstate_per_qd):
		ham_sys_AU[qd1_start + i, qd2_start + j] = ham_sys_AU[qd1_start + i, qd2_start + j] + (dseD[i] * dseA[j]) / E_cavity
for i in range(nstate_per_qd):
	for j in range(nstate_per_qd):
		ham_sys_AU[qd2_start + i, qd1_start + j] = ham_sys_AU[qd2_start + i, qd1_start + j] + (dseA[i] * dseD[j]) / E_cavity
for i in range(nstate_per_qd):
	for j in range(nstate_per_qd):
		ham_sys_AU[qd2_start + i, qd2_start + j] = ham_sys_AU[qd2_start + i, qd2_start + j] + (dseA[i] * dseA[j]) / E_cavity
#####################DSE####################


 
# ============================================================
# Build QD-local baths: each QD gets its own local phonon block.
# Cavity states remain phonon-free.
# ============================================================
nw = N_QD * nmode_per_qd # total QD phonon mode

# Build per-QD lists for vibrational modes and coupling blocks by repeating
# Tile the sorted blocks across QDs in each layer
w_AU = np.concatenate((w_AU_block1,w_AU_block2),axis=0)

Vklq_blocks = [Vklq_AU_block1, Vklq_AU_block2]

Vklq_AU = np.zeros((nw, ns, ns), dtype=float)
mode_owner = np.zeros(nw, dtype=np.int32)

# Assign modes and couplings for each QD
for i_qd in range(N_QD):
	V_block = Vklq_blocks[i_qd]

	# Mode and state ranges for this QD
	m0 = i_qd * nmode_per_qd
	m1 = m0 + nmode_per_qd
	s0 = i_qd * nstate_per_qd
	s1 = s0 + nstate_per_qd

	Vklq_AU[m0:m1, s0:s1, s0:s1] = V_block
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
