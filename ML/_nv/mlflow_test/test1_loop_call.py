import subprocess

import pathlib
curr_dir = pathlib.Path(__file__).parent.resolve() #pathlib.Path().resolve()

alphas = [1/x for x in range(1,10)]
l1_ratios = [1/x for x in range(1,10)]

for alpha in alphas:
    for l1 in l1_ratios:
        cmd = f"python {curr_dir}/test1.py {alpha} {l1}"

        returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
        print('returned value:', returned_value)