SSH stuff and commands:

run job: sbatch job.sh

check queue: squeue -u rasmus1142

Show job stuff more detailed: scontrol show job <jobID>

Kill job: scancel <jobID>







Dedicated interactive GPU node for testing (solve cursor/local running): idev -p gh-dev -t 2:00:00
