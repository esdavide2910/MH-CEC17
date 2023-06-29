 #!/bin/bash

rm -r results_SSA_py
mkdir results_SSA_py
python ./SSA.py
cd results_SSA_py
python ../extract.py
