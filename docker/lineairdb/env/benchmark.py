import re
import sys
import os

class Parameter:
  def __init__(self, name, values, default, dynamic = True):
    self.name = name
    self.values = values
    self.default = default
    self.dynamic = dynamic

def main():
  workload = "a"
  cpu = getNumCore(server)
  ram = getMemSize(server)
  for i, p1 in enumerate([parameters[0]]):
    for j, p2 in enumerate(parameters):
      if i >= j: continue
      for val1 in p1.values:
        for val2 in p2.values:
          if is_over_spec_limit(p1, val1, cpu, ram) or is_over_spec_limit(p2, val2, cpu, ram):
            continue
          args = set_args(p1, val1, p2, val2)
          arg = " ".join(args)
          for i in range(1):
            os.system("bench/ycsb -d 5000 -H true -l true -C 0.99 -c Silo -w {} -R 100000 {}".format(workload, arg))
            e = 40
            c = 30
            r = 0.8
            p = 3
            if p1.name == "prefetch_locality":
              p = val1
            elif p2.name == "prefetch_locality":
              p = val2
            if p1.name == "epoch":
              e = val1
            elif p2.name == "epoch":
              e = val2
            if p1.name == "checkpoint_interval":
              c = val1
            elif p2.name == "checkpoint_interval":
              c = val2
            if p1.name == "rehash_threshold":
              r = val1
            elif p2.name == "rehash_threshold":
              r = val2

            os.system('cat ycsb_result.json | jq -r "[.workload,.protocol,.threads,.clients,.handler,.tps,.commits,.aborts,.aborts+.commits,.etime,{e},{c},{r},{p}]|@csv" >> "/home/{server}-result.csv"'.format(server=server, e=e, c=c, r=r, p=p))
            os.system('echo "  \033[1;34m Attempt {} in 3 has done. \033[0m"'.format(i+1))
          os.system('echo -e "\n\n\n"')

def is_over_spec_limit(param, value, cpu, ram):
  # if param.name == "clients":
  #   return value > cpu
  # else:
    return False

def set_args(p1, value1, p2, value2):
  args = []
  for p in parameters:
    if p.name == p1.name:
      if p1.name != "prefetch_locality":
        args.append("--{name}={value1}".format(name=p.name, value1=value1))
      else:
        os.system("PREFETCH_LOCALITY={value1} cmake3 ../../LineairDB -DCMAKE_BUILD_TYPE=Release".format(value1=value1))
    elif p.name == p2.name:
      if p2.name != "prefetch_locality":
        args.append("--{name}={value2}".format(name=p.name, value2=value2))
      else:
        os.system("PREFETCH_LOCALITY={value2} cmake3 ../../LineairDB -DCMAKE_BUILD_TYPE=Release".format(value2=value2))
    elif p.name == "prefetch_locality":
      os.system("PREFETCH_LOCALITY={default} cmake3 ../../LineairDB -DCMAKE_BUILD_TYPE=Release".format(default=p.default))
    else:
      args.append("--{name}={default}".format(name=p.name, default=p.default))
  return args

def getNumCore(spec_string):
  return int(re.search(r'\d+', spec_string).group())

def getMemSize(spec_string):
  return int(re.search(r'\d+c(\d+)g', spec_string).group(1))

if __name__ == "__main__":
  argvs = sys.argv
  server = argvs[1]
  
  parameters = [
    Parameter("clients", [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 1, False),
    Parameter("prefetch_locality", [0, 1, 2, 3], 3, False),
    Parameter("epoch", [1, 10, 20, 30, 40], 40, False),
    Parameter("checkpoint_interval", [1, 10, 20, 30], 30), # discrete option
    Parameter("rehash_threshold", [0.1, 0.4, 0.8, 0.99], 0.8),
  ]
  
  main()