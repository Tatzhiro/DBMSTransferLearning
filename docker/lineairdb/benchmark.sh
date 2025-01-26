#!/bin/bash

servers=("4c6g" "8c12g" "12c16g" "16c24g" "24c32g")

for i in 1; do
  server=${servers[$i]}
  sed -i.bu "s/^server=.*/server="\"$server\""/" ./env/build.sh
  sed -i.bu "s/^server=.*/server="\"$server\""/" ./env/run.sh
  docker compose -f compose.yaml up $server -d
  docker exec $server chmod u+x /home/build.sh
  docker exec $server chmod u+x /home/run.sh
  docker exec $server chmod u+x /home/bench.sh
  docker exec $server /home/bench.sh &
  wait
  docker compose -f compose.yaml down $server
  docker volume prune -f
done
