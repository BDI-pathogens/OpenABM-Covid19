
# Compile the model
(cd src; make clean; make all)

python python/transpose_parameters.py

(cd src; ./covid19ibm.exe)
