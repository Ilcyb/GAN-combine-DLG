from multiprocessing import Process
import subprocess
import os
import time
import json

def read_experiment_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print('{} not exists'.format(config_path))
        exit(-1)

config_files = []
config_file_base_path = '../experiments_configs'

def execute_exp(config_file_path):
    config = read_experiment_config(config_file_path)
    print(config)
    subprocess.call('python -u main.py --config-file={} --mode=production --base-config=../experiments_configs/base_config.json | tee ../logs/{}.log'.format(config_file_path, config['name']), shell=True)

if __name__ == '__main__':

    for file in os.listdir(config_file_base_path):
        new_config = os.path.join(config_file_base_path, file)
        filename, file_ext = os.path.splitext(new_config)
        if file_ext != '.json' or 'base_config' in filename:
            continue
        config_files.append(new_config)

    process_list = []
    for i in range(len(config_files)):
        p = Process(target=execute_exp, args=(config_files[i], ))
        p.start()
        process_list.append(p)

    print('共开启{}个进程'.format(len(process_list)))
    start_time = time.time()

    for p in process_list:
        p.join()

    total_time = time.time() - start_time
    print(total_time)
    hours = int(total_time / 3600)
    mins = int((total_time - (hours*3600)) / 60)
    secs = int((total_time - (hours*3600) - (mins*60)))

    print('所有进程执行完毕，共耗时{}小时{}分{}秒'.format(hours, mins, secs))