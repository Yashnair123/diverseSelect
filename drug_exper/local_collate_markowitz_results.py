import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

print("Starting")


ppf_df_array_dacs_val = []
ppf_df_array_jc_val = []
ppf_df_array_setting = []
ppf_df_array_alpha = []
ppf_df_array_gamma = []

gammas = ['gamma=0.015', 'gamma=0.02', 'gamma=0.025']
gamma_vals = [0.015, 0.02, 0.025]
alpha100 = 50
quantile_indexer = 1

mt_metric_fdr_power_label = []
mt_metric_rate = []
mt_metric_serr = []
mt_metric_alpha = []
mt_metric_setting = []
mt_metric_method = []
mt_metric_gamma = []

mt_metric_numr_method = []
mt_metric_numr_label = []
mt_metric_numr_rate = []
mt_metric_numr_serr = []
mt_metric_numr_alpha = []
mt_metric_numr_setting = []
mt_metric_numr_gamma = []


timeR = []
timeLabelR = []
serrR = []
settingR = []
alphaR = []
gammaR = []


max_time = -np.inf
bad_jobs = dict()
for setting in [0,1]:
    for couple in [True]:
        for alpha_ind in range(3):
            bad_jobs[(setting,couple,alpha_ind)] = []

for gamma_ind in range(3):
    for couple in [True]:
        results = \
            pd.read_csv(f'collated_markowitz_results/metrics_c{couple}_q{quantile_indexer}_g{gamma_ind}_a{alpha100}.csv', header=None).to_numpy().astype('float')
        vanilla_results = \
            pd.read_csv(f'collated_markowitz_results/vanilla_metrics_c{couple}_q{quantile_indexer}_g{gamma_ind}_a{alpha100}.csv', header=None).to_numpy().astype('float')
        
        print("Loading histogram results...")
        histogram_results = []

        with open(f'collated_markowitz_results/histogram_c{couple}_q{quantile_indexer}_g{gamma_ind}_a{alpha100}.csv', 'r') as file:
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

            result = results[job][:-2]
            vanilla_result = vanilla_results[job]
            

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
            

            histogram_result = np.array(histogram_results[job])

            ppf = np.sum(histogram_result <= diversity_result)/len(histogram_result)
            ppf_df_array_dacs_val.append(ppf)
            

            vanilla_ppf = np.sum(histogram_result <= vanilla_diversity_result)/len(histogram_result)
            ppf_df_array_jc_val.append(vanilla_ppf)

            
            ppf_df_array_gamma.append(gammas[gamma_ind])

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
            
            mt_metric_method.append('DACS')
            mt_metric_gamma.append(gammas[gamma_ind])

            mt_metric_fdr_power_label.append(['FDR', 'Power'][label_ind])
            mt_metric_rate.append([np.mean(mt_metric_vanilla_fdr_arr), \
                                np.mean(mt_metric_vanilla_power_arr)][label_ind])
            mt_metric_serr.append([np.std(mt_metric_vanilla_fdr_arr)/np.sqrt(len(mt_metric_vanilla_fdr_arr)), \
                                np.std(mt_metric_vanilla_power_arr)/np.sqrt(len(mt_metric_vanilla_power_arr))][label_ind])
            
            mt_metric_method.append('CS')
            mt_metric_gamma.append(gammas[gamma_ind])



        mt_metric_numr_method.append('DACS')
        mt_metric_numr_label.append('#R')
        mt_metric_numr_rate.append(np.mean(mt_metric_dacs_numr_arr))
        mt_metric_numr_serr.append(np.std(mt_metric_dacs_numr_arr)/np.sqrt(len((mt_metric_dacs_numr_arr))))
        
        mt_metric_numr_gamma.append(gammas[gamma_ind])


        mt_metric_numr_method.append('CS')
        mt_metric_numr_label.append('#R')
        mt_metric_numr_rate.append(np.mean(mt_metric_vanilla_numr_arr))
        mt_metric_numr_serr.append(np.std(mt_metric_vanilla_numr_arr)/np.sqrt(len((mt_metric_vanilla_numr_arr))))
        
        mt_metric_numr_gamma.append(gammas[gamma_ind])


        timeR.append(np.mean(time_arr))
        timeLabelR.append('Time (sec)')
        serrR.append(np.std(time_arr)/np.sqrt(len(time_arr)))
        
        gammaR.append(gammas[gamma_ind])




mark_ppf_results = pd.DataFrame({
    'DACS': ppf_df_array_dacs_val,
    'CS': ppf_df_array_jc_val,
    'gamma': ppf_df_array_gamma
})

mark_ppf_results.to_csv('./markowitz_csvs_to_plot/markowitz_ppf_results.csv')


mark_mt_metric_results = pd.DataFrame({
    'method': mt_metric_method,
    'serr': mt_metric_serr,
    'error_metric': mt_metric_fdr_power_label,
    'rate': mt_metric_rate,
    'gamma': mt_metric_gamma
})

mark_mt_metric_results.to_csv('./markowitz_csvs_to_plot/markowitz_mt_metric_results.csv')


mark_mt_numr_results = pd.DataFrame({
    'method': mt_metric_numr_method,
    'serr': mt_metric_numr_serr,
    'num_rejections': mt_metric_numr_label,
    '#R': mt_metric_numr_rate,
    'gamma': mt_metric_numr_gamma
})

mark_mt_numr_results.to_csv('./markowitz_csvs_to_plot/markowitz_mt_numr_results.csv')


mark_time_results = pd.DataFrame({
    'Time (sec)': timeR,
    'time': timeLabelR,
    'serr': serrR,
    'gamma': gammaR
})

mark_time_results.to_csv('./markowitz_csvs_to_plot/markowitz_time_results.csv')

print(f'max time: {np.max(timeR)}')