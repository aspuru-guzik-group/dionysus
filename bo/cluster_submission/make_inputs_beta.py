#!/usr/bin/env python

import os, sys

def make_job_dir(filename):

	split = filename.split('-')
	assert len(split)==6
	dataset = split[0]
	goal = split[1]
	model = split[2]
	feature_type = split[3]
	acq_func_type = split[4]
	beta = split[5]

	print('beta :', beta)

	replace_dict = {
		'DATASET': dataset, 'GOAL': goal, 
		'MODEL': model, 'FEATURE': feature_type, 
		'ACQ_FUNC': acq_func_type, 'BETA': beta
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
	'delaney-minimize-gp-mfp-ucb-1.0',
	'delaney-minimize-gp-mfp-ucb-0.75',
	'delaney-minimize-gp-mfp-ucb-0.5',
	'delaney-minimize-gp-mfp-ucb-0.25',
	'delaney-minimize-gp-mfp-ucb-0.0',

	'delaney-minimize-gp-mordred-ucb-1.0',
	'delaney-minimize-gp-mordred-ucb-0.75',
	'delaney-minimize-gp-mordred-ucb-0.5',
	'delaney-minimize-gp-mordred-ucb-0.25',
	'delaney-minimize-gp-mordred-ucb-0.0',
	
	# freesolv
	'freesolv-minimize-gp-mfp-ucb-1.0',
	'freesolv-minimize-gp-mfp-ucb-0.75',
	'freesolv-minimize-gp-mfp-ucb-0.5',
	'freesolv-minimize-gp-mfp-ucb-0.25',
	'freesolv-minimize-gp-mfp-ucb-0.0',

	'freesolv-minimize-gp-mordred-ucb-1.0',
	'freesolv-minimize-gp-mordred-ucb-0.75',
	'freesolv-minimize-gp-mordred-ucb-0.5',
	'freesolv-minimize-gp-mordred-ucb-0.25',
	'freesolv-minimize-gp-mordred-ucb-0.0',
]


for job in to_make:
	make_job_dir(job)
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