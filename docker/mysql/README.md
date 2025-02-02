# MySQL benchmark

## Requirement

1. Docker
2. Python
3. sysbench 1.1 (and Percona lab TPC-C)

## Install sysbench 1.1

sysbench will be installed in /usr/local/bin

### Ubuntu

```bash
sudo apt -y install make automake libtool pkg-config libaio-dev
sudo apt -y install libmysqlclient-dev libssl-dev
git clone https://github.com/akopytov/sysbench.git
cd sysbench
./autogen.sh
./configure
make -j
sudo make install
sudo sysbench --version
cd ..
git clone https://github.com/Percona-Lab/sysbench-tpcc.git
```

### MacOS

Requires MySQL version < 8.3

```bash
brew install automake libtool openssl pkg-config
# install MySQL 8.2
# brew uninstall mysql-client
curl https://raw.githubusercontent.com/Homebrew/homebrew-core/2f35529519fb6a2cc361ce3d464a1bd181505a54/Formula/m/mysql-client.rb -o mysql-client.rb
brew install ./mysql-client.rb
export LIBRARY_PATH=$LIBRARY_PATH:$(brew --prefix zstd)/lib/
export LIBRARY_PATH=$LIBRARY_PATH:/opt/homebrew/opt/openssl/lib/
git clone https://github.com/akopytov/sysbench.git
cd sysbench
./autogen.sh
./configure
make -j
sudo make install
sudo sysbench --version
```

## How to run

```bash
cd docker/mysql
nohup python benchmark.py &
```

## Output

result/{Machine}-result.csv
