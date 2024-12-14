# snellius introduction

## Submit slurm job

Schedule job

        sbatch slurm.sh

You can see the status of your job with the following command

        squeue

When the job is running, a output file will spawn called: jupyter_%x_%j.out. Read it with:

        cat jupyter_%x_%j.out

