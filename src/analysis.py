import re
import os
import json
import sys
from utils import read_experiment_config
from PIL import Image
import torchvision.transforms.functional as TF

experiment_name_re_1 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>[a-z]+)$')
experiment_name_re_2 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>.+)_nr=(?P<norm_rate>[0-9e-]+)$')

psnr_re = re.compile(r'iter-\d+:([0-9\.-]+)')

result_img_re = re.compile(r'result1-(?P<batch_num>\d)\.png')
truth_img_re = re.compile(r'truth_img1-(?P<batch_num>\d)\.png')

class AnalysisFolder:

    def __init__(self, root_path, folder_name_regexs, analysis_method, merge_method) -> None:
        self.root_path = root_path
        self.folder_name_regexs = folder_name_regexs
        self.analysis_method = analysis_method
        self.merge_method = merge_method
        self.working_list = []

    def scan(self):
        for sub_file_path in os.listdir(self.root_path):
            complete_sub_file_path = os.path.join(self.root_path, sub_file_path)
            # print(complete_sub_file_path)
            if not os.path.isdir(complete_sub_file_path):
                continue
            sub_analysis_folder = AnalysisFolder(complete_sub_file_path, self.folder_name_regexs, 
                                    self.analysis_method, self.merge_method)
            for regex in self.folder_name_regexs:
                regex_match = regex.match(complete_sub_file_path)
                if regex_match is not None:
                    self.working_list.append(sub_analysis_folder)
                    break
            sub_analysis_folder.scan()
            self.working_list += sub_analysis_folder.working_list

    
    def analysis(self, **kwargs):
        results = []
        for folder in self.working_list:
            results.append(self.analysis_method(folder, **kwargs))
        return self.merge_method(results, **kwargs)
        

def AnalysisMeanPSNR(log_path):
    psnr = []
    with open(log_path, 'r') as f:
        for line in f.readlines():
            psnr.append(float(psnr_re.match(line)[1]))
    max_psnr = 0
    mean_psnr = 0
    for p in psnr:
        if p > max_psnr:
            max_psnr = p
        mean_psnr += p
    mean_psnr = mean_psnr / len(psnr)
    return max_psnr, mean_psnr

def AnalysisPermutationsPSNR(folder_path):
    file_list = os.listdir(folder_path)
    result_img_list = []
    truth_img_list = []
    for filename in file_list:
        result_match = result_img_re.match(filename)
        truth_match = truth_img_re.match(filename)
        if result_match is not None:
            result_img = Image.open(os.path.join(folder_path, filename))
            result_img = TF.to_tensor(result_img)
            result_img_list.append(result_img)
        elif truth_match is not None:
            truth_img = Image.open(os.path.join(folder_path, filename))
            truth_img = TF.to_tensor(truth_img)
            truth_img_list.append(truth_img)
        else:
            continue
    
def CommonExperimentsAnalysis(folder : AnalysisFolder, **kwargs):
    current_path = folder.root_path
    regex_result = None
    for regex in folder.folder_name_regexs:
        regex_result = regex.match(current_path)
        if regex_result is None:
            continue
        else:
            break
    attributions = regex_result.groupdict()
    training_folders_paths = []
    for sub_file in os.listdir(current_path):
        sub_file_path = os.path.join(current_path, sub_file)
        if os.path.isdir(sub_file_path):
            training_folders_paths.append(sub_file_path)
    
    total_mean_psnr = 0
    topK_psnr = {}
    for training_path in training_folders_paths:
        try:
            max_psnr, mean_psnr = AnalysisMeanPSNR(os.path.join(training_path, 'meanpsnr.log'))
            total_mean_psnr += mean_psnr
            topK_psnr["file://"+os.path.join(training_path, 'compare.png')] = max_psnr
        except FileNotFoundError:
            print(os.path.join(training_path, 'meanpsnr.log') + ' not found')
    total_mean_psnr = total_mean_psnr / len(training_folders_paths)
    topK_psnr = {k:v for k, v in sorted(topK_psnr.items(), key=lambda x:x[1], reverse=True)[:kwargs['maxTopK']]}
    
    print('{} Analysis complete'.format(current_path))
    return dict(attributions=attributions, topK_psnr=topK_psnr, mean_psnr=total_mean_psnr)

def MegerCommonExperimentsAnalysis(sub_results:list, **kwargs):
    # 同dataset, batch_size, init_method, norm, norm_rate
    results = {}
    for sub_result in sub_results:
        attributions = sub_result['attributions']
        attributions['norm_rate'] = attributions.get('norm_rate', 'None')
        exp_combine_key = kwargs['exp_combine_key'].format(**attributions)
        if results.get(exp_combine_key) is None:
            results[exp_combine_key] = {}
            # results[exp_combine_key]['attributions'] = attributions
            results[exp_combine_key]['mean_psnr'] = sub_result['mean_psnr']
            results[exp_combine_key]['mean_psnr_count'] = 1
            results[exp_combine_key]['topK_psnr'] = sub_result['topK_psnr']
            continue
        results[exp_combine_key]['mean_psnr'] += sub_result['mean_psnr']
        results[exp_combine_key]['mean_psnr_count'] += 1
        results[exp_combine_key]['topK_psnr'] = {**(results[exp_combine_key]['topK_psnr']), **sub_result['topK_psnr']}
    
    for key, result in results.items():
        result['mean_psnr'] = result['mean_psnr'] / result['mean_psnr_count']
        result['topK_psnr'] = {k:v for k, v in sorted(result['topK_psnr'].items(), key=lambda x:x[1], reverse=True)[:kwargs['maxTopK']]} 
        del(result['mean_psnr_count'])
        results[key] = result

    return results

def main(config_path):
    configs = read_experiment_config(config_path)
    for analysis_config in configs['analysis_configs']:
        analysis_folder = AnalysisFolder(analysis_config['data_path'], [experiment_name_re_1, experiment_name_re_2],
            CommonExperimentsAnalysis, MegerCommonExperimentsAnalysis)
        analysis_folder.scan()
        result = analysis_folder.analysis(**analysis_config['analysis_config'])
        with open(analysis_config['output_path'], 'w') as f:
            json.dump(result, f, ensure_ascii=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('请指定分析配置文件')
    main(sys.argv[1])
