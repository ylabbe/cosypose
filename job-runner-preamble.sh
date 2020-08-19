source $CONDA_ROOT/bin/activate
conda activate $CONDA_ENV
cd $PROJECT_DIR

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
