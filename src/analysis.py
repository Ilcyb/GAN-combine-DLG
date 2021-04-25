import re
import os
import json
import sys
import pathlib
import uuid
from urllib.parse import urlparse, unquote
from utils import read_experiment_config, calculate_psnr
from PIL import Image
import torchvision.transforms.functional as TF

# GAN-combine-DLG regex
experiment_name_re_1 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>[a-z]+)$')
experiment_name_re_2 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>.+)_nr=(?P<norm_rate>[0-9e-]+)$')
experiment_name_re_3 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>.+)_sd-(?P<smooth_direction>.+)_nr-(?P<norm_rate>[0-9e-]+)$')
experiment_name_re_4 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>[a-z]+)_sd-(?P<smooth_direction>[0-9])$')
experiment_name_re_6 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>.+)_sd-(?P<smooth_direction>.+)_nr-(?P<norm_rate>[0-9e-]+)_nt-(?P<noise_type>.+)_nv-(?P<noise_variance>[\.0-9e-]+)$')
experiment_name_re_7 = re.compile(r'.+ds-(?P<dataset>.+)_bs-(?P<batch_size>.+)_init-(?P<init_method>.+)_iter-(?P<iters>.+)_op-(?P<optim>.+)_nm-(?P<norm>.+)_sd-(?P<smooth_direction>.+)_nr-(?P<norm_rate>[0-9e-]+)_nt-(?P<noise_type>[a-z]+)$')

# inverting-gradients regex
experiment_name_re_5 = re.compile(r'.+name_.+_ds-(?P<dataset>.+)_bs-(?P<batch_size>[0-9]+)$')

blacklist = ['procedure']

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
            black_flag = False
            for black_word in blacklist:
                if black_word in sub_file_path:
                    black_flag = True
                    break
            if black_flag == True:
                continue
            complete_sub_file_path = os.path.join(self.root_path, sub_file_path)
            # print(complete_sub_file_path)
            if not os.path.isdir(complete_sub_file_path):
                continue
            sub_analysis_folder = AnalysisFolder(complete_sub_file_path, self.folder_name_regexs, 
                                    self.analysis_method, self.merge_method)
            if sub_analysis_folder.isWorkingFolder():
                self.working_list.append(sub_analysis_folder)
            sub_analysis_folder.scan()
            self.working_list += sub_analysis_folder.working_list

    
    def analysis(self, **kwargs):
        results = []
        for folder in self.working_list:
            results.append(self.analysis_method(folder, **kwargs))
        return self.merge_method(results, **kwargs)

    def isWorkingFolder(self):
        for regex in self.folder_name_regexs:
            regex_match = regex.match(self.root_path)
            if regex_match is not None:
                return True
        return False

def AnalysisMeanPSNR(folder_path):
    log_path = os.path.join(folder_path, 'meanpsnr.log')
    psnr = []
    with open(log_path, 'r') as f:
        for line in f.readlines():
            psnr.append(float(psnr_re.match(line)[1]))
    max_psnr = 0
    for p in psnr:
        if p > max_psnr:
            max_psnr = p
    return max_psnr

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

    if len(result_img_list) == 0:
        raise FileNotFoundError('{} experiment is not finished'.format(folder_path))

    max_total_psnr = 0
    for result_img in result_img_list:
        max_psnr = 0
        for truth_img in truth_img_list:
            psnr = calculate_psnr(result_img, truth_img).item()
            if psnr > max_psnr:
                max_psnr = psnr
        max_total_psnr += max_psnr
    
    return max_total_psnr/len(result_img_list)
    
def GetPSNRList(folder_path):
    log_path = os.path.join(folder_path, 'meanpsnr.log')
    psnr = []
    with open(log_path, 'r') as f:
        for line in f.readlines():
            psnr.append(float(psnr_re.match(line)[1]))
    return psnr

# inverting gradients 实验获取PSNR方法
def GetPSNRByJsonFile(folder_path):
    info_path = os.path.join(folder_path, 'info.json')
    with open(info_path, 'r') as f:
        info = json.load(f)
        return info['psnr']
    
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
    training_nums = 0
    for training_path in training_folders_paths:
        try:
            max_psnr = kwargs['psnr_extract_method'](training_path)
            total_mean_psnr += max_psnr
            topK_psnr[pathlib.Path(os.path.join(training_path, 'compare.png')).as_uri()] = max_psnr
            training_nums+=1
        except FileNotFoundError as e:
            print(str(e))
    try:
        total_mean_psnr = total_mean_psnr / training_nums
        topK_psnr = {k:v for k, v in sorted(topK_psnr.items(), key=lambda x:x[1], reverse=True)[:kwargs['maxTopK']]}
    except ZeroDivisionError:
        print('{} is empty'.format(current_path))
    else:
        print('{} Analysis complete'.format(current_path))
    
    return dict(attributions=attributions, topK_psnr=topK_psnr, mean_psnr=total_mean_psnr, training_nums=training_nums)

def MegerCommonExperimentsAnalysis(sub_results:list, **kwargs):
    # 同dataset, batch_size, init_method, norm, norm_rate
    results = {}
    for sub_result in sub_results:
        attributions = sub_result['attributions']
        attributions['norm_rate'] = attributions.get('norm_rate', 'None')
        attributions['smooth_direction'] = attributions.get('smooth_direction', 'None')
        attributions['noise_type'] = attributions.get('noise_type', 'None')
        attributions['noise_variance'] = attributions.get('noise_variance', 'None')
        exp_combine_key = kwargs['exp_combine_key'].format(**attributions)
        if results.get(exp_combine_key) is None:
            results[exp_combine_key] = {}
            # results[exp_combine_key]['attributions'] = attributions
            results[exp_combine_key]['mean_psnr'] = sub_result['mean_psnr']
            results[exp_combine_key]['mean_psnr_count'] = 1
            results[exp_combine_key]['topK_psnr'] = sub_result['topK_psnr']
            results[exp_combine_key]['training_nums'] = sub_result['training_nums']
            continue
        results[exp_combine_key]['mean_psnr'] += sub_result['mean_psnr']
        results[exp_combine_key]['mean_psnr_count'] += 1
        results[exp_combine_key]['topK_psnr'] = {**(results[exp_combine_key]['topK_psnr']), **sub_result['topK_psnr']}
        results[exp_combine_key]['training_nums'] += sub_result['training_nums']
    
    for key, result in results.items():
        result['mean_psnr'] = result['mean_psnr'] / result['mean_psnr_count']
        result['topK_psnr'] = {k:v for k, v in sorted(result['topK_psnr'].items(), key=lambda x:x[1], reverse=True)[:kwargs['maxTopK']]} 
        del(result['mean_psnr_count'])
        results[key] = result

    return results

def CompareImageAnalysis(folder : AnalysisFolder, **kwargs):
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
    for sub_file in os.listdir(folder.root_path):
        sub_file_path = os.path.join(folder.root_path, sub_file)
        if os.path.isdir(sub_file_path):
            training_folders_paths.append(sub_file_path)
    
    compare_imgs_paths = []
    for training_path in training_folders_paths:
        compare_img_path = os.path.join(training_path, 'compare.png')
        if not os.path.exists(compare_img_path):
            print('{} don\'t have a compare.png'.format(training_path))
            continue
        compare_imgs_paths.append(compare_img_path)
    print('{} Analysis complete'.format(current_path))
    return dict(attributions=attributions, compare_imgs_paths=compare_imgs_paths)

def MergeCompareImageAnalysis(sub_results:list, **kwargs):
    results = {}
    for sub_result in sub_results:
        attributions = sub_result['attributions']
        attributions['norm_rate'] = attributions.get('norm_rate', 'None')
        attributions['smooth_direction'] = attributions.get('smooth_direction', 'None')
        attributions['noise_type'] = attributions.get('noise_type', 'None')
        attributions['noise_variance'] = attributions.get('noise_variance', 'None')
        exp_combine_key = kwargs['exp_combine_key'].format(**attributions)
        if results.get(exp_combine_key) is None:
            results[exp_combine_key] = {}
            results[exp_combine_key] = sub_result['compare_imgs_paths']
            continue
        results[exp_combine_key] += sub_result['compare_imgs_paths']
    
    return results

def PSNRListAnalysis(folder : AnalysisFolder, **kwargs):
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
    
    training_nums = 0
    psnr_list_len = kwargs['psnr_list_len']
    total_psnr_list = [0] * psnr_list_len
    for training_path in training_folders_paths:
        try:
            psnr_list = kwargs['psnr_extract_method'](training_path)
            if len(psnr_list) < psnr_list_len:
                psnr_list += [psnr_list[-1]] * (psnr_list_len-len(psnr_list))
            elif len(psnr_list) > psnr_list_len:
                psnr_list = psnr_list[:psnr_list_len]
            total_psnr_list = [i+j for i,j in zip(total_psnr_list, psnr_list)]
            training_nums+=1
        except FileNotFoundError as e:
            print(str(e))

    print('{} Analysis complete'.format(current_path))
    return dict(attributions=attributions, training_nums=training_nums, total_psnr_list=total_psnr_list)

def MergePSNRListAnalysis(sub_results:list, **kwargs):
    results = {}
    for sub_result in sub_results:
        attributions = sub_result['attributions']
        attributions['norm_rate'] = attributions.get('norm_rate', 'None')
        attributions['smooth_direction'] = attributions.get('smooth_direction', 'None')
        attributions['noise_type'] = attributions.get('noise_type', 'None')
        attributions['noise_variance'] = attributions.get('noise_variance', 'None')
        exp_combine_key = kwargs['exp_combine_key'].format(**attributions)
        if results.get(exp_combine_key) is None:
            results[exp_combine_key] = {}
            results[exp_combine_key]['training_nums'] = sub_result['training_nums']
            results[exp_combine_key]['total_psnr_list'] = sub_result['total_psnr_list']
            results[exp_combine_key]['mean_psnr_list'] = [0] * len(sub_result['total_psnr_list'])
            continue
        results[exp_combine_key]['training_nums'] += sub_result['training_nums']
        results[exp_combine_key]['total_psnr_list'] = [i+j for i,j in zip(sub_result['total_psnr_list'], results[exp_combine_key]['total_psnr_list'])]
    
    for key, result in results.items():
        if result['training_nums'] == 0:
            pass
        else:
            result['mean_psnr_list'] = [psnr/result['training_nums'] for psnr in result['total_psnr_list']]
        del(result['total_psnr_list'])
        results[key] = result

    return results

def main(config_path):
    configs = read_experiment_config(config_path)
    for analysis_config in configs['analysis_configs']:
        if analysis_config['enable'] == False:
            continue
        if analysis_config['experiment_type'] == 'GAN-combine-DLG':
            regexs = [experiment_name_re_1, experiment_name_re_2, 
            experiment_name_re_3, experiment_name_re_4, experiment_name_re_6, experiment_name_re_7]
        elif analysis_config['experiment_type'] == 'inverting-gradients':
            regexs = [experiment_name_re_5]

        if analysis_config['analysis_method'] == 'psnr':
            analysis_folder = AnalysisFolder(analysis_config['data_path'], regexs,
                CommonExperimentsAnalysis, MegerCommonExperimentsAnalysis)
            analysis_folder.scan()
            kwargs = analysis_config['analysis_config']
            if kwargs['psnr_extract'] == 'common':
                kwargs['psnr_extract_method'] = AnalysisMeanPSNR
            elif kwargs['psnr_extract'] == 'permutations':
                kwargs['psnr_extract_method'] = AnalysisPermutationsPSNR
            elif kwargs['psnr_extract'] == 'from-json-file':
                kwargs['psnr_extract_method'] = GetPSNRByJsonFile
            result = analysis_folder.analysis(**kwargs)
            with open(analysis_config['output_path'], 'w') as f:
                json.dump(result, f, ensure_ascii=False, allow_nan=False)
        elif analysis_config['analysis_method'] == 'compare_img':
            analysis_folder = AnalysisFolder(analysis_config['data_path'], regexs,
                CommonExperimentsAnalysis, MegerCommonExperimentsAnalysis)
            analysis_folder.scan()
            kwargs = analysis_config['analysis_config']
            if kwargs['psnr_extract'] == 'common':
                kwargs['psnr_extract_method'] = AnalysisMeanPSNR
            elif kwargs['psnr_extract'] == 'permutations':
                kwargs['psnr_extract_method'] = AnalysisPermutationsPSNR
            result = analysis_folder.analysis(**kwargs)
            if not os.path.exists(analysis_config['output_path']):
                os.mkdir(analysis_config['output_path'])
            for folder_name, result_psnr in result.items():
                count = 1
                folder_path = os.path.join(analysis_config['output_path'], folder_name)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                for compare_img_path, _ in sorted(result_psnr['topK_psnr'].items(), key=lambda x:x[1], reverse=True)[:kwargs['maxTopK']]:
                    compare_img_path = urlparse(unquote(compare_img_path, 'utf-8'))
                    compare_img_path = os.path.abspath(os.path.join(compare_img_path.netloc, compare_img_path.path))
                    os.symlink(compare_img_path, os.path.join(folder_path, 'compare_{}.png'.format(count)))
                    count += 1
        elif analysis_config['analysis_method'] == 'psnr_list':
            analysis_folder = AnalysisFolder(analysis_config['data_path'], regexs,
                PSNRListAnalysis, MergePSNRListAnalysis)
            analysis_folder.scan()
            kwargs = analysis_config.get('analysis_config')
            kwargs['psnr_extract_method'] = GetPSNRList
            result = analysis_folder.analysis(**kwargs)
            with open(analysis_config['output_path'], 'w') as f:
                json.dump(result, f, ensure_ascii=False, allow_nan=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('请指定分析配置文件')
    main(sys.argv[1])
