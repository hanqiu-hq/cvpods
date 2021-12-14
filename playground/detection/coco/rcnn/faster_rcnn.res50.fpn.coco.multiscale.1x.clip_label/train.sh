#!/bin/bash

srun -p shlab_cv_gp --gres=gpu:8 --quotatype=reserve --cpus-per-task=96 pods_train --num-gpus 8