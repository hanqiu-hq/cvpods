#!/bin/bash

srun -p shlab_cv_gp --gres=gpu:4 --quotatype=reserved --cpus-per-task=48 pods_train --num-gpus 4