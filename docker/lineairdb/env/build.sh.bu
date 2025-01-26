#!/bin/bash

#
# Install prerequisites and build LineairDB
# 

repository="https://github.com/LineairDB/LineairDB.git"
server="8c12g"

function build(){
  FILE="lineairdb/build/bench/ycsb"
  if [ ! -f "$FILE" ]; then
    echo "File $FILE does not exist."
    rm -rf lineairdb

    # REQUIREMENTS
    sudo yum install -y epel-release
    sudo yum install -y --enablerepo=epel git cmake3 numactl-devel libatomic jq make gcc-c++ libstdc++-static
    sudo yum install -y centos-release-scl
    sudo yum install -y devtoolset-11
    source /opt/rh/devtoolset-11/enable

    if [ ! -e "/lib64/libatomic.so" ]; then
      sudo ln -s /lib64/libatomic.so{.1,}
    fi

    # BUILD
    if [ ! -d "lineairdb" ]; then
      git clone --recursive $repository
    fi
    
    if [ ! -f "$FILE" ]; then
      rm -rf lineairdb/build
      mkdir -p lineairdb/build; cd $_;
      cmake3 ../../LineairDB -DCMAKE_BUILD_TYPE=Release
      make -j $(grep -c ^processor /proc/cpuinfo)
    fi
  else
    echo "File $FILE exists."
  fi
}

if [ "$server" = "" ]; then
    echo "Error: please modify server name"
    exit
fi

echo "Repository: $repository"
echo "Server: $server build"
mkdir -p result
build | tee "$(dirname $0)/result/$server-build.log"

echo "Build done"
