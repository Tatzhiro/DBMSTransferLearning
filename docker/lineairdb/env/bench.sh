#!/bin/bash

cd /home
rm -rf lineairdb
rm -rf LineairDB
./build.sh
./run.sh &
wait 
echo "bench.sh done"