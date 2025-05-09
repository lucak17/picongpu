#!/bin/bash -l
# Copyright 2025 Luca Pennati
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#


## PIConGPU batch script for dardel's SLURM batch system

#SBATCH -A !TBG_nameProject
#SBATCH -p !TBG_queue
#SBATCH --time=!TBG_wallTime
# Sets batch job's name
#SBATCH --job-name=!TBG_jobName

#SBATCH --ntasks=!TBG_tasks
###SBATCH --mem=400G 
#SBATCH --cpus-per-task=!TBG_coresPerGPU
#SBATCH --ntasks-per-gpu=1
#SBATCH --gpu-bind=closest
#SBATCH --mail-type=!TBG_mailSettings
#SBATCH --mail-user=!TBG_mailAddress
#SBATCH --chdir=!TBG_dstPath
#SBATCH --exclusive
#SBATCH -o stdout
#SBATCH -e stderr


echo "Start calculation tbg"

## calculations will be performed by tbg ##
.TBG_queue="gpu"

## settings that can be controlled by environment variables before submit
.TBG_mailSettings=${MY_MAILNOTIFY:-"NONE"}
.TBG_mailAddress=${MY_MAIL:-"someone@example.com"}
.TBG_author=${MY_NAME:+--author \"${MY_NAME}\"}
.TBG_nameProject=${PROJID:-""}
.TBG_profile=${PIC_PROFILE:-"~/picongpu.profile"}

## number of available/hosted devices per node in the system
.TBG_numHostedDevicesPerNode=8

# host memory per device
.TBG_memPerDevice=60000

# number of CPU cores to block per GPU
# we have 8 CPU cores per GPU (64cores/8gpus ~ 8cores)
# but one core (= core 0) is reserved for system processes
# accordign to OLCF docs
.TBG_coresPerGPU=7

# required GPUs per node for the current job
.TBG_devicesPerNode=$(if [ $TBG_tasks -gt $TBG_numHostedDevicesPerNode ] ; then echo $TBG_numHostedDevicesPerNode; else echo $TBG_tasks; fi)

# use ceil to caculate nodes
.TBG_nodes="$((( TBG_tasks + TBG_devicesPerNode - 1 ) / TBG_devicesPerNode))"

## end calculations ##

echo 'Start job with !TBG_nodes nodes. '

cd !TBG_dstPath

export MODULES_NO_OUTPUT=1
#source !TBG_profile
if [ $? -ne 0 ] ; then
    echo "Error: PIConGPU environment profile under \"!TBG_profile\" not found!"
    exit 1
fi
unset MODULES_NO_OUTPUT

# set user rights to u=rwx;g=r-x;o=---
umask 0027

mkdir simOutput 2> /dev/null
cd simOutput
ln -s ../stdout output

# number of broken nodes
n_broken_nodes=0

# return code of cuda_memcheck
node_check_err=0

export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1

if [ $node_check_err -eq 0 ]  ; then
    # Run PIConGPU
    echo "Start PIConGPU."
    date
    echo "Nodes: !TBG_nodes Tasks:!TBG_tasks"
    #test $n_broken_nodes -ne 0 && exclude_nodes="-x./bad_nodes.txt"
    #srun -n !TBG_tasks --nodes=!TBG_nodes $exclude_nodes -K1 !TBG_dstPath/input/bin/picongpu --mpiDirect !TBG_author !TBG_programParams
    srun -n !TBG_tasks --nodes=!TBG_nodes  -K1 !TBG_dstPath/input/bin/picongpu  --mpiDirect !TBG_programParams
    echo "End PIConGPU."
else
    echo "Job stopped because of previous issues."
    echo "Job stopped because of previous issues." >&2
fi
