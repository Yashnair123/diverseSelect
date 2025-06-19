import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

print("Starting")

fdp_dict = dict()
tdp_dict = dict()
num_rejections_dict = dict()
diversity_dict = dict()
time_dict = dict()

ppf_dict = dict()

dicts = [fdp_dict, tdp_dict, num_rejections_dict, diversity_dict, time_dict]

vanilla_fdp_dict = dict()
vanilla_tdp_dict = dict()
vanilla_num_rejections_dict = dict()
vanilla_diversity_dict = dict()

vanilla_ppf_dict = dict()

vanilla_dicts = [vanilla_fdp_dict, vanilla_tdp_dict, vanilla_num_rejections_dict, \
                 vanilla_diversity_dict]

for setting in [0, 1]:
    for couple in [True, False]:
        for alpha_ind in range(3):
            for d in dicts:
                d[(couple,setting,alpha_ind)] = []
            for vd in vanilla_dicts:
                vd[(couple,setting,alpha_ind)] = []

            ppf_dict[(couple,setting,alpha_ind)] = []
            vanilla_ppf_dict[(couple,setting,alpha_ind)] = []


ppf_df_array_dacs_val = []
ppf_df_array_jc_val = []
ppf_df_array_setting = []
ppf_df_array_alpha = []
alphas = ['alpha=0.05', 'alpha=0.2', 'alpha=0.35']



mt_metric_fdr_power_label = []
mt_metric_rate = []
mt_metric_serr = []
mt_metric_alpha = []
mt_metric_setting = []
mt_metric_method = []

mt_metric_numr_method = []
mt_metric_numr_label = []
mt_metric_numr_rate = []
mt_metric_numr_serr = []
mt_metric_numr_alpha = []
mt_metric_numr_setting = []


timeR = []
timeLabelR = []
serrR = []
settingR = []
alphaR = []

bad_jobs = dict()
for setting in [0,1]:
    for couple in [True]:
        for alpha_ind in range(3):
            bad_jobs[(setting,couple,alpha_ind)] = []

for alpha_ind in range(3):
    for setting in [0,1]:
        
        for couple in [True]:
            results = \
                pd.read_csv(f'collated_sharpe_results/metrics_c{couple}_s{setting}_a{alpha_ind}.csv', header=None).to_numpy().astype('float')
            
            vanilla_results = \
                pd.read_csv(f'collated_sharpe_results/vanilla_metrics_c{couple}_s{setting}_a{alpha_ind}.csv', header=None).to_numpy().astype('float')
            
            
            print("Loading histogram results...")
            histogram_results = []

            with open(f'collated_sharpe_results/histogram_c{couple}_s{setting}_a{alpha_ind}.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    histogram_results.append([float(num) for num in row])


            ppfs = []
            vanilla_ppfs = []
            data = []

            num_rejections = []
            vanilla_num_rejections = []

            mt_metric_dacs_fdr_arr = []
            mt_metric_vanilla_fdr_arr = []

            mt_metric_dacs_power_arr = []
            mt_metric_vanilla_power_arr = []

            mt_metric_dacs_numr_arr = []
            mt_metric_vanilla_numr_arr = []

            time_arr = []
            for job in tqdm(range(250)):
                ppf_df_array_alpha.append(alphas[alpha_ind])
                result = results[job][:-2]
                vanilla_result = vanilla_results[job]
                
                for i in range(len(result)):
                    dicts[i][(couple,setting,alpha_ind)].append(result[i] if result[i] != -np.inf else 0.)
                for i in range(len(vanilla_result)):
                    vanilla_dicts[i][(couple,setting,alpha_ind)].append(vanilla_result[i] if vanilla_result[i] != -np.inf else 0.)

                mt_metric_dacs_fdr_arr.append(result[0])
                mt_metric_vanilla_fdr_arr.append(vanilla_result[0])

                mt_metric_dacs_power_arr.append(result[1])
                mt_metric_vanilla_power_arr.append(vanilla_result[1])

                mt_metric_dacs_numr_arr.append(result[2])
                mt_metric_vanilla_numr_arr.append(vanilla_result[2])
                diversity_result = result[-2]
                vanilla_diversity_result = vanilla_result[-1]
                

                diversity_numr = result[-3]
                vanilla_numr = vanilla_result[-2]
                
                time_arr.append(result[-1])
                

                histogram_result = histogram_results[job]


                ppf = np.sum(histogram_result <= diversity_result)/len(histogram_result)
                ppf_df_array_dacs_val.append(ppf)

                
                ppf_df_array_setting.append(['Setting 1', 'Setting 2'][setting])

                num_rejections.append(num_rejections_dict[(couple,setting,alpha_ind)][job])
                vanilla_num_rejections.append(vanilla_num_rejections_dict[(couple,setting,alpha_ind)][job])

                vanilla_ppf = np.sum(histogram_result <= vanilla_diversity_result)/len(histogram_result)
                ppf_df_array_jc_val.append(vanilla_ppf)
                

                ppfs.append(ppf)
                vanilla_ppfs.append(vanilla_ppf)

                if vanilla_ppf > ppf:
                    bad_jobs[(setting,couple,alpha_ind)].append(job)

            for label_ind in range(2):
                mt_metric_fdr_power_label.append(['FDR', 'Power'][label_ind])
                mt_metric_rate.append([np.mean(mt_metric_dacs_fdr_arr), \
                                       np.mean(mt_metric_dacs_power_arr)][label_ind])
                mt_metric_serr.append([np.std(mt_metric_dacs_fdr_arr)/np.sqrt(len(mt_metric_dacs_fdr_arr)), \
                                       np.std(mt_metric_dacs_power_arr)/np.sqrt(len(mt_metric_dacs_power_arr))][label_ind])
                mt_metric_alpha.append(alphas[alpha_ind])
                mt_metric_setting.append(['Setting 1', 'Setting 2'][setting])
                mt_metric_method.append('DACS')

                mt_metric_fdr_power_label.append(['FDR', 'Power'][label_ind])
                mt_metric_rate.append([np.mean(mt_metric_vanilla_fdr_arr), \
                                       np.mean(mt_metric_vanilla_power_arr)][label_ind])
                mt_metric_serr.append([np.std(mt_metric_vanilla_fdr_arr)/np.sqrt(len(mt_metric_vanilla_fdr_arr)), \
                                       np.std(mt_metric_vanilla_power_arr)/np.sqrt(len(mt_metric_vanilla_power_arr))][label_ind])
                mt_metric_alpha.append(alphas[alpha_ind])
                mt_metric_setting.append(['Setting 1', 'Setting 2'][setting])
                mt_metric_method.append('CS')



            mt_metric_numr_method.append('DACS')
            mt_metric_numr_label.append('#R')
            mt_metric_numr_rate.append(np.mean(mt_metric_dacs_numr_arr))
            mt_metric_numr_serr.append(np.std(mt_metric_dacs_numr_arr)/np.sqrt(len((mt_metric_dacs_numr_arr))))
            mt_metric_numr_alpha.append(alphas[alpha_ind])
            mt_metric_numr_setting.append(['Setting 1', 'Setting 2'][setting])


            mt_metric_numr_method.append('CS')
            mt_metric_numr_label.append('#R')
            mt_metric_numr_rate.append(np.mean(mt_metric_vanilla_numr_arr))
            mt_metric_numr_serr.append(np.std(mt_metric_vanilla_numr_arr)/np.sqrt(len((mt_metric_vanilla_numr_arr))))
            mt_metric_numr_alpha.append(alphas[alpha_ind])
            mt_metric_numr_setting.append(['Setting 1', 'Setting 2'][setting])


            timeR.append(np.mean(time_arr))
            timeLabelR.append('Time (sec)')
            serrR.append(np.std(time_arr)/np.sqrt(len(time_arr)))
            settingR.append(['Setting 1', 'Setting 2'][setting])
            alphaR.append(alphas[alpha_ind])


sharpe_ppf_results = pd.DataFrame({
    'DACS': ppf_df_array_dacs_val,
    'Setting': ppf_df_array_setting,
    'CS': ppf_df_array_jc_val,
    'alpha': ppf_df_array_alpha
})

sharpe_ppf_results.to_csv('./sharpe_csvs_to_plot/sharpe_ppf_results.csv')


sharpe_mt_metric_results = pd.DataFrame({
    'method': mt_metric_method,
    'Setting': mt_metric_setting,
    'serr': mt_metric_serr,
    'alpha': mt_metric_alpha,
    'error_metric': mt_metric_fdr_power_label,
    'rate': mt_metric_rate
})

sharpe_mt_metric_results.to_csv('./sharpe_csvs_to_plot/sharpe_mt_metric_results.csv')


sharpe_mt_numr_results = pd.DataFrame({
    'method': mt_metric_numr_method,
    'Setting': mt_metric_numr_setting,
    'serr': mt_metric_numr_serr,
    'alpha': mt_metric_numr_alpha,
    'num_rejections': mt_metric_numr_label,
    '#R': mt_metric_numr_rate
})

sharpe_mt_numr_results.to_csv('./sharpe_csvs_to_plot/sharpe_mt_numr_results.csv')


sharpe_time_results = pd.DataFrame({
    'Time (sec)': timeR,
    'time': timeLabelR,
    'serr': serrR,
    'setting': settingR,
    'alpha_var': alphaR
})

sharpe_time_results.to_csv('./sharpe_csvs_to_plot/sharpe_time_results.csv')