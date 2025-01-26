#!/bin/bash

docker run --gpus all -it -v /raid/tatsuhironm/dml:/home --name dml nvcr.io/nvidia/pytorch:23.09-py3
docker exec -it metric_learning /bin/bash