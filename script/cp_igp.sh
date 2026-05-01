#!/bin/bash

rsync -av --whole-file dataset/lotte/igp swarm:/mnt/nfs/home/juelinliu/scratch1/dataset-gpu-mvr/dataset/lotte
rsync -av --whole-file dataset/hotpot/igp swarm:/mnt/nfs/home/juelinliu/scratch1/dataset-gpu-mvr/dataset/hotpot
rsync -av --whole-file dataset/msmarco/igp swarm:/mnt/nfs/home/juelinliu/scratch1/dataset-gpu-mvr/dataset/msmarco
