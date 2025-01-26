import subprocess
from time import sleep, time
import re
import datetime
import os

class Parameter:
  def __init__(self, name: str, values: list, default, dynamic: bool = True) -> None:
    self.name: str = name
    self.values: list = values
    self.default = default
    self.dynamic: bool = dynamic

def main():
  print(datetime.datetime.now(), flush=True)
  global cnt
  subprocess.run(["git", "restore", "mysql-docker-compose.yaml"])
  set_default_configuration()
  for cpu, ram in servers:
    cnt = 0
    set_docker_hardware_resources(cpu, ram)

    benchmark(cpu, ram)
    
    print(datetime.datetime.now(), flush=True)
    print(f"benchmark for {cpu}c{ram}g finished with {cnt} measurements", flush=True)

def benchmark(cpu, ram):
  server = f"{cpu}c{ram}g"
  global cnt

  for num_table in num_tables:
    for table_size in table_sizes:
      initialize_mysql()
      assert(subprocess.run(mysql_connection + ["-e", "drop database if exists sbtest; create database sbtest"]).returncode == 0)
      assert(subprocess.run(["sudo", "sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-user=root", 
                            f"--tables={num_table}", f"--table-size={table_size}", f"--threads={os.cpu_count()}", 
                            "oltp_common", "prepare"]).returncode == 0)

      for workload in workloads:

        sysbench_settings = [workload, num_table, table_size, num_client]
        if not os.path.isfile(f"result/{server}-result.csv"):
          csv_clms = [param.name for param in parameters] + ["tps"] + ["workload", "num_table", "table_size", "num_client"]
          csv_clms = ",".join(csv_clms) + "\n"
          with open(f"result/{server}-result.csv", "w") as f:
            f.write(csv_clms)
        assert_default_configuration()

        for i, param1 in enumerate(parameters):
          for j, param2 in enumerate(parameters):
            if j <= i: continue

            for value1 in param1.values:
              if is_over_spec_limit(param1, value1, cpu, ram):
                continue
              for value2 in param2.values:
                if is_over_spec_limit(param2, value2, cpu, ram):
                  continue
                if is_violation_of_constraint(param1, value1, param2, value2, cpu, ram):
                  continue


                subprocess.run('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"', shell=True)

                start = time() 
                
                set_configuration(param1, param2, value1, value2)

                result = subprocess.run(["sudo", "sysbench", "--db-driver=mysql", "--mysql-host=127.0.0.1", "--mysql-user=root",  
                                        f"--tables={num_table}", f"--table-size={table_size}", f"--threads={num_client}", 
                                        "--warmup-time=30", f"--time={run_time}", workload, "run"], capture_output=True, text=True).stdout
                
                # purge binlog
                assert(subprocess.run(mysql_connection + ["-e", "reset master"]).returncode == 0)

                row = make_row(param1, param2, value1, value2, result, sysbench_settings)
                with open(f"result/{server}-result.csv", "a") as f:
                  f.write(row)

                end = time()
                cnt += 1
                print(end - start, flush=True)
                print(cnt, flush=True)
                
            set_default_configuration()
            

def is_over_spec_limit(param: Parameter, value, cpu, ram):
  if param.name == "innodb_buffer_pool_size" and int(''.join(filter(str.isdigit, value))) > 0.75 * ram:
    return True
  if param.name == "innodb_write_io_threads" and value >= cpu:
    return True
  if param.name == "innodb_read_io_threads" and value >= cpu:
    return True
  
  return False
  
def is_violation_of_constraint(param1: Parameter, value1, param2: Parameter, value2, cpu, ram):
  """Check whether parameters violate user-defined constraints"""
  
  if param1.name == "innodb_write_io_threads" and param2.name == "innodb_read_io_threads":
    return (value1 + value2) > cpu
  if param1.name == "innodb_read_io_threads" and param2.name == "innodb_write_io_threads":
    return (value1 + value2) > cpu
  
  if param1.name == "innodb_write_io_threads" or param1.name == "innodb_read_io_threads":
    return (value1 + param1.default) > cpu
  if param2.name == "innodb_write_io_threads" or param2.name == "innodb_read_io_threads":
    return (value2 + param2.default) > cpu

  return False
  


def set_docker_hardware_resources(cpu: int, ram: int):
  subprocess.run(['sed', '-i', f"s/^          cpus:.*/          cpus: \'{cpu}\'/", "./mysql-docker-compose.yaml"])
  subprocess.run(['sed', '-i', f"s/^          memory:.*/          memory: {ram}gb/", "./mysql-docker-compose.yaml"])

def assert_default_configuration():
  with open("./mysql-docker-compose.yaml", "r") as f:
    lines = f.read().splitlines()

  for line in lines:
    for parameter in parameters:
      if parameter.name in line:
        assert(str(parameter.default) in line)
        break

def extract_throughput(result):
  pattern = r"transactions:\s+\d+\s+\((\d+\.\d+) per sec\.\)"
  match = re.search(pattern, result)
  return match.group(1)

def make_row(param1, param2, value1, value2, sysbench_result, sysbench_settings):
  """Format database parameters, sysbench result, and sysbench settings into a row of csv"""

  throughput = extract_throughput(sysbench_result)

  row = []
  for param in parameters:
    if param.name == param1.name:
      row.append(str(value1))
    elif param.name == param2.name:
      row.append(str(value2))
    else:
      row.append(str(param.default))
  row.append(str(throughput))

  for clm in sysbench_settings:
    row.append(str(clm))
  return ",".join(row) + "\n"
    

def set_default_configuration():
  for parameter in parameters:
    write_mycnf(parameter.name, parameter.default)
  restart_mysql()

def set_configuration(param1, param2, value1, value2):
  if not param1.dynamic or not param2.dynamic:
    write_mycnf(param1.name, value1)
    write_mycnf(param2.name, value2)
    restart_mysql()
  else:
    assert(subprocess.run(mysql_connection + ["-e", f"SET GLOBAL {param1.name}={value1}"]).returncode == 0)
    assert(subprocess.run(mysql_connection + ["-e", f"SET GLOBAL {param2.name}={value2}"]).returncode == 0)


def write_mycnf(name, value):
  assert(subprocess.run(['sed', '-i', f"s/^      - --{name}=.*/      - --{name}={value}/", "./mysql-docker-compose.yaml"]).returncode == 0)

def restart_mysql():
  assert(subprocess.run(["sudo", "docker", "compose", "-f", "mysql-docker-compose.yaml", "stop", "mysql"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
  assert(subprocess.run(["sudo", "docker", "compose", "-f", "mysql-docker-compose.yaml", "up", "mysql", "-d"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
  wait_connection()

def initialize_mysql():
  assert(subprocess.run(["sudo", "docker", "compose", "-f", "mysql-docker-compose.yaml", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
  subprocess.run(["sudo", "docker", "volume", "prune", "-f"])
  assert(subprocess.run(["sudo", "docker", "compose", "-f", "mysql-docker-compose.yaml", "up", "mysql", "-d"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
  wait_connection()

def wait_connection():
  while True:
    mysql_is_up = (subprocess.run(mysql_connection + ["-e", "SELECT 1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
    if mysql_is_up: return

# ログ用
def calculate_iter(parameters: list):
  iter = 0
  for i, param1 in enumerate(parameters):
    for j, param2 in enumerate(parameters):
      if j <= i: continue
      iter += len(param1.values) * len(param2.values)
  return iter

if __name__ == "__main__":
  # ログ用
  cnt = 0

  # ワークロード設定
  num_tables = [64]
  table_sizes = [1000000, 5000000]
  run_time = 60
  num_client = 4
  workloads = ["oltp_read_write"]

  # 計測したいマシンのスペック: list[（CPU, RAM）]
  servers = [(12, 16), (24, 32)]

  # mysql接続用コマンド
  mysql_connection = ["mysql", "-u", "root", "-h", "127.0.0.1", "--port", "3306"]

  # レプリカ構成用パラメータ
  replication_parameters = [
    # Parameter("binlog_format", ["ROW", "MIXED", "STATEMENT"], "ROW"), # deprecated
    Parameter("binlog_row_image", ["full", "minimal", "noblob"], "full"),
    Parameter("binlog_transaction_compression", ["OFF", "ON"], "OFF"),
    # Parameter("rpl_semi_sync_slave_enabled", ["OFF", "ON"], "OFF"), # Unknown system variable 'rpl_semi_sync_slave_enabled'
    Parameter("slave_parallel_type", ["DATABASE", "LOGICAL_CLOCK"], "LOGICAL_CLOCK"),
    Parameter("slave_parallel_workers", [4], 4)
  ]
  
  # パラメータ
  parameters = [
    Parameter("innodb_buffer_pool_size", ["1GB", "2GB", "3GB", "4GB", "6GB", "8GB", "9GB", "12GB", "16GB", "18GB", "24GB"], "1GB", False), # in GB
    Parameter("innodb_read_io_threads", [1, 2, 3, 7, 15, 23], 2, False),
    Parameter("innodb_write_io_threads", [1, 2, 3, 7, 15, 23], 2, False),
    Parameter("innodb_flush_log_at_trx_commit", [0, 1, 2], 1), # discrete option
    Parameter("innodb_adaptive_hash_index", ["ON", "OFF"], "ON"),
    Parameter("sync_binlog", [0, 1], 1),
    Parameter("innodb_lru_scan_depth", [100, 1024, 5000, 10000], 1024),
    Parameter("innodb_buffer_pool_instances", [1, 2, 4, 8], "1", False), # The number of regions that the InnoDB buffer pool is divided into
    Parameter("innodb_change_buffer_max_size", [0, 10, 25, 50], 25), # percentage of the total size of the buffer pool
    Parameter("innodb_io_capacity", [100, 5000, 10000, 20000], 100), # number of I/O operations per second (IOPS) available to InnoDB background tasks
    Parameter("innodb_log_file_size", ["4MB", "48MB", "512MB", "5GB"], "48MB", False),
    Parameter("table_open_cache", [1, 1000, 2000, 4000], 4000),
  ]
  
  # レプリケーション構成ありの場合、以下のコードを実行
  # parameters = parameters + replication_parameters

  main()