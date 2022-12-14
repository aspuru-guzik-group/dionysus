#!/usr/bin/env python

import os, sys

def make_job_dir(filename):

	split = filename.split('-')
	assert len(split)==5
	dataset = split[0]
	goal = split[1]
	model = split[2]
	feature_type = split[3]
	acq_func_type = split[4]

	replace_dict = {
		'DATASET': dataset, 'GOAL': goal, 
		'MODEL': model, 'FEATURE': feature_type, 
		'ACQ_FUNC': acq_func_type
	}

	with open('submit_template.sh', 'r') as content:
		lines = content.readlines()

	new_lines = []
	for line in lines:
		for key, val in replace_dict.items():
			line = line.replace(key, val)
		new_lines.append(line)

	os.makedirs(filename, exist_ok=True)

	with open(f'{filename}/submit.sh', 'w') as f:
		for line in new_lines:
			f.write(line)

	os.system(f'cp run.py {filename}/')






#------------
# REGRESSION
#------------

to_make = [
	# delaney
	'delaney-minimize-ngboost-mfp-ucb',
	'delaney-minimize-ngboost-mordred-ucb',
	'delaney-minimize-ngboost-graphembed-ucb',
	'delaney-minimize-gp-mfp-ucb',
	'delaney-minimize-gp-mordred-ucb',
	'delaney-minimize-gp-graphembed-ucb',
	'delaney-minimize-bnn-mfp-ucb',
	'delaney-minimize-bnn-mordred-ucb',
	'delaney-minimize-bnn-graphembed-ucb',
	'delaney-minimize-sngp-mfp-ucb',
	'delaney-minimize-sngp-mordred-ucb',
	'delaney-minimize-sngp-graphembed-ucb',
	'delaney-minimize-gnngp-graphnet-ucb',
	# freesolv
	'freesolv-minimize-ngboost-mfp-ucb',
	'freesolv-minimize-ngboost-mordred-ucb',
	'freesolv-minimize-ngboost-graphembed-ucb',
	'freesolv-minimize-gp-mfp-ucb',
	'freesolv-minimize-gp-mordred-ucb',
	'freesolv-minimize-gp-graphembed-ucb',
	'freesolv-minimize-bnn-mfp-ucb',
	'freesolv-minimize-bnn-mordred-ucb',
	'freesolv-minimize-bnn-graphembed-ucb',
	'freesolv-minimize-sngp-mfp-ucb',
	'freesolv-minimize-sngp-mordred-ucb',
	'freesolv-minimize-sngp-graphembed-ucb',
	'freesolv-minimize-gnngp-graphnet-ucb',
]


make_job_dir(to_make[0])



#--------
# BINARY
#--------


"""
#!/bin/bash
#
#SBATCH -J delaney-gp-mfp-ucb
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task 4
#SBATCH --time=200:00:00
#SBATCH --partition=cpu
#SBATCH --qos=nopreemption
#SBATCH --export=ALL
#SBATCH --output=gpmol.log
#SBATCH --gres=gpu:0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/h/rhickman/sw/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/h/rhickman/sw/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/h/rhickman/sw/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/h/rhickman/sw/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate gpmol

date >> gpmol.log
echo "" >> gpmol.log
python run.py delaney minimize bnn graphembed ucb
echo "" >> gpmol.log
date >> gpmol.log

"""