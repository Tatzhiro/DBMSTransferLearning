#!/bin/bash

#
# Benchmarking with LineairDB's YCSB executable
# Requirement: lineairdb/build.sh must be executed before this script
#

max_epoch=40
server="8c12g"

function task(){
  FILE="lineairdb/build/bench/ycsb"
  if [ -f "$FILE" ]; then
  :
  else
      echo "Error: the benchmark executable $FILE does not exist."
      echo "Please run build.sh first to generate the executable."
      exit 1
  fi

  echo "workload,protocol,thread,clients,handler,tps,commits,aborts,aborts+commits,elapsed_time,epoch_duration,checkpoint_interval,rehash_threshold,prefetch_locality" > /home/$server-result.csv
  cd lineairdb/build

  python /home/benchmark.py $server
}

# BENCHMARK
if [ "$server" = "" ]; then
    echo "Error: please modify server name"
    exit
fi

echo "Benchmark Started"
mkdir -p result
task > "$(dirname 0)/result/$server.log" 2>&1 &
wait