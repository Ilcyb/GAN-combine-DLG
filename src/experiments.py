from multiprocessing import Process
import subprocess
import os
import time

config_files = []
config_file_base_path = '../experiments_configs'

for file in os.listdir(config_file_base_path):
    new_config = os.path.join(config_file_base_path, file)
    filename, file_ext = os.path.splitext(new_config)
    if file_ext != '.json':
        continue
    config_files.append(new_config)

def execute_exp(config_file_path):
    subprocess.call('python main.py --config-file={} --mode=production --device=cpu --base-config=../experiments_configs/base_config.json'.format(config_file_path))

process_list = []
for i in range(len(config_files)):
    p = Process(target=execute_exp, args=(config_files[i]))
    p.start()
    process_list.append(p)

print('共开启{}个进程'.format(len(process_list)))
start_time = time.time()

for p in process_list:
    p.join()

total_time = time.time() - start_time
hours = total_time % 3600
mins = (total_time - (hours*3600)) % 60
secs = (total_time - (hours*3600) - (mins*60))

print('所有进程执行完毕，共耗时{}小时{}分{}秒'.format(hours, mins, secs))