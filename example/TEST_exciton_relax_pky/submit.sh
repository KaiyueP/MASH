#!/bin/bash
#SBATCH -J mash-real-test 
#SBATCH --qos=debug
#SBATCH --time=00:30:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaiyue_peng@berkeley.edu

export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

HomeDir=$(pwd)
ScrDir=/pscratch/sd/k/kaiyuep/mash_test/LVC_exciton_relax_pky
mkdir -p "$ScrDir"

cp mash.py lvc_ext.in lvc_params_AU.npz "$ScrDir/"
cd "$ScrDir/"

module load python
module load conda
conda activate mash

python mash.py @lvc_ext.in -ckpt -ckptfile test_real_ckpt.npz -ckptfrac 0.01 >  run.dat

cp run.dat pop.out "$HomeDir/"


