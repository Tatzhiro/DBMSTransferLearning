# MySQL benchmark

## Requirement
1. Docker
2. Python
3. sysbench 1.1

## Install sysbench 1.1 on ubuntu
sysbench will be installed in /usr/local/bin
```
sudo apt -y install make automake libtool pkg-config libaio-dev
sudo apt -y install libmysqlclient-dev libssl-dev
git clone https://github.com/akopytov/sysbench.git
cd sysbench
./autogen.sh
./configure
make -j
sudo make install
sudo sysbench --version
```

## How to run
```
cd docker/mysql
nohup python benchmark.py &
```

## Output
result/{Machine}-result.csv