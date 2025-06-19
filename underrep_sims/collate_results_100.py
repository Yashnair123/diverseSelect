import numpy as np
import pandas as pd

variants = []
full_results = dict()
full_count_results = dict()
for ml_alg_ind in range(3):
    for setting in range(6):
        for alpha_ind in range(3):
            variants.append((ml_alg_ind, setting, alpha_ind))
            for regime in ['dacs', 'bh', 'env']:
                full_results[( ml_alg_ind,\
                            setting, alpha_ind, regime)]\
                    = []
                full_count_results[( ml_alg_ind,\
                            setting, alpha_ind, regime)]\
                    = []
for variant in range(54):
    (ml_alg_ind, setting, alpha_ind) = variants[variant]

    count_data = pd.read_csv(f'results_100/counts_v{variant}.csv', header=None).to_numpy().astype('float')
    if len(count_data) == 0:
        print(f'bad variant: {variant}')
    total_freq_arr = []
    total_freq = np.zeros(count_data.shape[1])
    total_count = []
    total_non_zero = 0
    total_zero = 0
    avg_diversity_metric = 0.
    avg_diversities = []
    for j in range(len(count_data)):
        total_count.append(count_data[j])
        if np.any(count_data[j]>0):
            total_freq += count_data[j]/np.sum(count_data[j])
            total_non_zero += 1
            diversity_metric = np.min(count_data[j])/np.sum(count_data[j])
            avg_diversities.append(diversity_metric)
            total_freq_arr.append(count_data[j]/np.sum(count_data[j]))
        else:
            total_zero += 1
            avg_diversities.append(-1./count_data.shape[1])
    total_count = np.array(total_count)
    if total_non_zero == 0:
        total_freq = np.zeros(len(count_data[j]))
    else:
        total_freq = total_freq / total_non_zero
    avg_diversity_metric = np.mean(avg_diversities)
    serr_diversity_metric = np.std(avg_diversities)/np.sqrt(len(avg_diversities))

    vanilla_count_data = pd.read_csv(f'results_100/vanilla_counts_v{variant}.csv', header=None).to_numpy().astype('float')
    vanilla_total_freq_arr = []
    vanilla_total_freq = np.zeros(vanilla_count_data.shape[1])
    vanilla_total_count = []
    vanilla_total_non_zero = 0
    vanilla_total_zero = 0
    vanilla_avg_diversities = []
    for j in range(len(vanilla_count_data)):
        vanilla_total_count.append(vanilla_count_data[j])
        if np.any(vanilla_count_data[j]>0):
            vanilla_total_freq += vanilla_count_data[j]/np.sum(vanilla_count_data[j])
            vanilla_total_non_zero += 1
            vanilla_diversity_metric = np.min(vanilla_count_data[j])/np.sum(vanilla_count_data[j])
            vanilla_avg_diversities.append(vanilla_diversity_metric)
            vanilla_total_freq_arr.append(vanilla_count_data[j]/np.sum(vanilla_count_data[j]))
        else:
            vanilla_total_zero += 1
            vanilla_avg_diversities.append(-1./(vanilla_count_data.shape[1]))
    vanilla_total_count = np.array(vanilla_total_count)
    if vanilla_total_non_zero == 0:
        vanilla_total_freq = np.zeros(len(vanilla_count_data[j]))
    else:
        vanilla_total_freq = vanilla_total_freq / vanilla_total_non_zero
    vanilla_avg_diversity_metric = np.mean(vanilla_avg_diversities)
    vanilla_serr_diversity_metric = np.std(vanilla_avg_diversities)/np.sqrt(len(vanilla_avg_diversities))

    metrics_data = pd.read_csv(f'results_100/metrics_v{variant}.csv', header=None).to_numpy().astype('float')
    vanilla_metrics_data = pd.read_csv(f'results_100/vanilla_metrics_v{variant}.csv', header=None).to_numpy().astype('float')
    avg_metrics_data = np.mean(metrics_data,axis=0)
    avg_vanilla_metrics_data = np.mean(vanilla_metrics_data,axis=0)
    serr_metrics_data = np.std(metrics_data,axis=0)/np.sqrt(metrics_data.shape[0])
    serr_vanilla_metrics_data = np.std(vanilla_metrics_data,axis=0)/np.sqrt(vanilla_metrics_data.shape[0])

    pi0s = pd.read_csv(f'results_100/pi0s_v{variant}.csv', header=None).to_numpy().astype('float')
    conditional_pi0s = pd.read_csv(f'results_100/conditional_pi0s_v{variant}.csv', header=None).to_numpy().astype('float')

    avg_pi0 = np.mean(pi0s)
    serr_pi0 = np.std(pi0s)/np.sqrt(len(pi0s))

    avg_conditional_pi0s = np.mean(conditional_pi0s, axis=0)
    serr_conditional_pi0s = np.std(conditional_pi0s, axis=0)/np.sqrt(conditional_pi0s.shape[0])

    
    full_results[(ml_alg_ind, setting, alpha_ind, 'dacs')] \
    = [total_freq, total_zero/len(count_data), [avg_diversity_metric, serr_diversity_metric],\
       avg_metrics_data, serr_metrics_data, np.array(total_freq_arr)]

    full_results[(ml_alg_ind, setting, alpha_ind, 'bh')] \
    = [vanilla_total_freq, vanilla_total_zero/len(vanilla_count_data), [vanilla_avg_diversity_metric, vanilla_serr_diversity_metric],\
       avg_vanilla_metrics_data, serr_vanilla_metrics_data, np.array(vanilla_total_freq_arr)]

    full_results[(ml_alg_ind, setting, alpha_ind, 'env')] \
    = [avg_pi0, serr_pi0, avg_conditional_pi0s, serr_conditional_pi0s]


    full_count_results[((ml_alg_ind, setting, alpha_ind, 'dacs'))] = [np.mean(total_count,axis=0), \
                                            np.std(total_count,axis=0)/np.sqrt(len(total_count))]
    full_count_results[((ml_alg_ind, setting, alpha_ind, 'bh'))] = [np.mean(vanilla_total_count,axis=0), \
                                    np.std(vanilla_total_count,axis=0)/np.sqrt(len(vanilla_total_count))]

x_noises = np.linspace(0.5, 1.5, 10)
settings = ['Setting 1', 'Setting 2', 'Setting 3', 'Setting 4', 'Setting 5', 'Setting 6']
ml_algs = ['OLS', 'MLP', 'SVM', 'Oracle']
alphas = ['alpha = 0.05', 'alpha = 0.2', 'alpha = 0.35']
###################################################################################################
for ml_alg_ind in range(3):
    diversityR = []
    alphaR = []
    serrR = []
    methodR = []
    settingR = []
    for setting in range(6):
        for vanilla_alpha_ind in range(3):
            for dacs_alpha_ind in range(3):
                [diversity_metric,serr_diversity_metric] = full_results[(ml_alg_ind, \
                                                setting, dacs_alpha_ind, 'dacs')][2]
            
                diversityR.append(diversity_metric)
                alphaR.append(alphas[dacs_alpha_ind])
                serrR.append(serr_diversity_metric)
                methodR.append('DACS')
                settingR.append(settings[setting])
            
                [vanilla_diversity_metric,vanilla_serr_diversity_metric] = full_results[(\
                                        ml_alg_ind, setting, vanilla_alpha_ind, 'bh')][2]
                diversityR.append(vanilla_diversity_metric)
                alphaR.append(alphas[vanilla_alpha_ind])
                serrR.append(vanilla_serr_diversity_metric)
                methodR.append('CS')
                settingR.append(settings[setting])

    diversity_dfR = pd.DataFrame({
        'underrep_ind': diversityR,
        'alpha': alphaR,
        "serr": serrR,
        'method': methodR,
        'diversity': ['underrep_ind']*len(methodR),
        'setting': settingR
    })

    diversity_dfR.to_csv(f'./csvs_to_plot_100/diversity_results_R_{ml_alg_ind}.csv')


# ###################################################################################################

for ml_alg_ind in range(3):
    freqsR = []
    clusterR = []
    methodR = []
    serrR = []
    alphaR = []
    settingR = []
    for setting in range(6):
        for dacs_alpha_ind in range(3):
            for vanilla_alpha_ind in range(3):
                [freq, prop0] = full_results[(ml_alg_ind, \
                                                    setting, dacs_alpha_ind, 'dacs')][:2]
                total_freq_arr = full_results[(ml_alg_ind, \
                                                    setting, dacs_alpha_ind, 'dacs')][-1]
                if len(total_freq_arr) == 0:
                    for c in range(len(freq)):
                        freqsR.append(0)
                        clusterR.append(c)
                        methodR.append("DACS")
                        serrR.append(0)
                        alphaR.append(alphas[dacs_alpha_ind])
                        settingR.append(settings[setting])
                else:
                    for c in range(len(freq)):
                        freqsR.append(freq[c])
                        clusterR.append(c)
                        methodR.append("DACS")
                        serrR.append(np.std(total_freq_arr,axis=0)[c]/np.sqrt(total_freq_arr.shape[0]))
                        alphaR.append(alphas[dacs_alpha_ind])
                        settingR.append(settings[setting])

                [vanilla_freq, vanilla_prop0] \
                    = full_results[(ml_alg_ind, \
                                                    setting, vanilla_alpha_ind, 'bh')][:2]
                vanilla_total_freq_arr = full_results[(ml_alg_ind, \
                                                    setting, vanilla_alpha_ind, 'bh')][-1]
                if len(vanilla_total_freq_arr) == 0:
                    for c in range(len(vanilla_freq)):
                        freqsR.append(0)
                        clusterR.append(c)
                        methodR.append("CS")
                        serrR.append(0)
                        alphaR.append(alphas[vanilla_alpha_ind])
                        settingR.append(settings[setting])
                else:
                    for c in range(len(vanilla_freq)):
                        freqsR.append(vanilla_freq[c])
                        clusterR.append(c)
                        methodR.append("CS")
                        serrR.append(np.std(vanilla_total_freq_arr,axis=0)[c]/np.sqrt(vanilla_total_freq_arr.shape[0]))
                        alphaR.append(alphas[vanilla_alpha_ind])
                        settingR.append(settings[setting])

    lines_prop0sR = []
    lines_sim_settingR = []
    lines_methodR = []
    lines_vanillaR = []

    for setting in range(6):
        for dacs_alpha_ind in range(3):
            for vanilla_alpha_ind in range(3):
                [freq, prop0] = full_results[(ml_alg_ind, \
                                                    setting, dacs_alpha_ind, 'dacs')][:2]
                lines_prop0sR.append(prop0)
                lines_sim_settingR.append(settings[setting])
                lines_methodR.append(f'{alphas[dacs_alpha_ind]}')
                lines_vanillaR.append('DACS')

                [vanilla_freq, vanilla_prop0] = full_results[(ml_alg_ind, \
                                                    setting, vanilla_alpha_ind, 'bh')][:2]
                lines_prop0sR.append(vanilla_prop0)
                lines_sim_settingR.append(settings[setting])
                lines_methodR.append(f'{alphas[vanilla_alpha_ind]}')
                lines_vanillaR.append('CS')

    df = pd.DataFrame({
        'proportions': freqsR,
        'cluster': clusterR,
        'alpha': alphaR,
        'serr': serrR,
        'method': methodR,
        'setting': settingR
    })

    df_lines = pd.DataFrame({
        'p0': lines_prop0sR,
        'setting': lines_sim_settingR,
        'alpha': lines_methodR,
        'method': lines_vanillaR
    })

    df.to_csv(f'./csvs_to_plot_100/cluster_results_R_{ml_alg_ind}.csv')
    df_lines.to_csv(f'./csvs_to_plot_100/line_results_R_{ml_alg_ind}.csv')
###################################################################################################
for ml_alg_ind in range(3):
    countsR = []
    clusterR = []
    methodR = []
    serrR = []
    alphaR = []
    settingR = []
    for setting in range(6):
        for dacs_alpha_ind in range(3):
            for vanilla_alpha_ind in range(3):
                [count, count_serr] = full_count_results[(ml_alg_ind, \
                                                    setting, dacs_alpha_ind, 'dacs')]
                
                for c in range(len(count)):
                    countsR.append(count[c])
                    clusterR.append(c)
                    methodR.append("DACS")
                    serrR.append(count_serr[c])
                    alphaR.append(alphas[dacs_alpha_ind])
                    settingR.append(settings[setting])

                [vanilla_count, vanilla_count_serr] = full_count_results[(ml_alg_ind, \
                                                    setting, vanilla_alpha_ind, 'bh')]
                for c in range(len(vanilla_count)):
                    countsR.append(vanilla_count[c])
                    clusterR.append(c)
                    methodR.append("CS")
                    serrR.append(vanilla_count_serr[c])
                    alphaR.append(alphas[vanilla_alpha_ind])
                    settingR.append(settings[setting])

    df = pd.DataFrame({
        'counts': countsR,
        'cluster': clusterR,
        'alpha': alphaR,
        'serr': serrR,
        'method': methodR,
        'setting': settingR
    })

    df.to_csv(f'./csvs_to_plot_100/count_results_R_{ml_alg_ind}.csv')
###################################################################################################
for ml_alg_ind in range(3):
    mt_metricR = []
    cluster_fdp_tdp_numrR = []
    methodR = []
    serrR = []
    alphaR = []
    settingR = []
    for setting in range(6):
        for dacs_alpha_ind in range(3):
            for vanilla_alpha_ind in range(3):
                mt_metrics = full_results[(ml_alg_ind,\
                                    setting, dacs_alpha_ind, 'dacs')][3]
                serr_metrics = full_results[(ml_alg_ind,\
                                    setting, dacs_alpha_ind, 'dacs')][4]
                for c in range(2):
                    mt_metricR.append(mt_metrics[c])
                    cluster_fdp_tdp_numrR.append(['FDR', 'Power', '#R'][c])
                    methodR.append("DACS")
                    serrR.append(serr_metrics[c])
                    alphaR.append(alphas[dacs_alpha_ind])
                    settingR.append(settings[setting])
                

                vanilla_mt_metrics = full_results[(ml_alg_ind,\
                                    setting, vanilla_alpha_ind, 'bh')][3]
                vanilla_serr_metrics = full_results[(ml_alg_ind,\
                                    setting, vanilla_alpha_ind, 'bh')][4]
                for c in range(2):
                    mt_metricR.append(vanilla_mt_metrics[c])
                    cluster_fdp_tdp_numrR.append(['FDR', 'Power', '#R'][c])
                    methodR.append("CS")
                    serrR.append(vanilla_serr_metrics[c])
                    alphaR.append(alphas[vanilla_alpha_ind])
                    settingR.append(settings[setting])


    mt_metric_dfR = pd.DataFrame({
        'rate': mt_metricR,
        'error_metric': cluster_fdp_tdp_numrR,
        'method': methodR,
        'serr': serrR,
        'alpha': alphaR,
        'setting': settingR
    })

    mt_metric_dfR.to_csv(f'./csvs_to_plot_100/mt_metric_results_R_{ml_alg_ind}.csv')
###################################################################################################
for ml_alg_ind in range(3):
    mt_metricR = []
    cluster_fdp_tdp_numrR = []
    methodR = []
    serrR = []
    alphaR = []
    settingR = []

    for setting in range(6):
        for dacs_alpha_ind in range(3):
            for vanilla_alpha_ind in range(3):
                mt_metrics = full_results[(ml_alg_ind,\
                                    setting, dacs_alpha_ind, 'dacs')][3]
                serr_metrics = full_results[(ml_alg_ind,\
                                    setting, dacs_alpha_ind, 'dacs')][4]
                for c in [2]:
                    mt_metricR.append(mt_metrics[c])
                    cluster_fdp_tdp_numrR.append(['FDR', 'Power', '#R'][c])
                    methodR.append("DACS")
                    serrR.append(serr_metrics[c])
                    alphaR.append(alphas[dacs_alpha_ind])
                    settingR.append(settings[setting])
                

                vanilla_mt_metrics = full_results[(ml_alg_ind,\
                                    setting, vanilla_alpha_ind, 'bh')][3]
                vanilla_serr_metrics = full_results[(ml_alg_ind,\
                                    setting, vanilla_alpha_ind, 'bh')][4]
                for c in [2]:
                    mt_metricR.append(vanilla_mt_metrics[c])
                    cluster_fdp_tdp_numrR.append(['FDR', 'Power', '#R'][c])
                    methodR.append("CS")
                    serrR.append(vanilla_serr_metrics[c])
                    alphaR.append(alphas[vanilla_alpha_ind])
                    settingR.append(settings[setting])


    num_rejects_dfR = pd.DataFrame({
        '#R': mt_metricR,
        'num_rejections': cluster_fdp_tdp_numrR,
        'method': methodR,
        'serr': serrR,
        'alpha': alphaR,
        'setting': settingR
    })

    num_rejects_dfR.to_csv(f'./csvs_to_plot_100/num_rejects_results_R_{ml_alg_ind}.csv')
###################################################################################################
timeR = []
timeLabelR = []
methodR = []
serrR = []
settingR = []
ml_algR = []
alphaR = []

for setting in range(6):
    for ml_alg_ind in range(3):

        for dacs_alpha_ind in range(3):
            time_avg = full_results[(ml_alg_ind,\
                                setting, dacs_alpha_ind, 'dacs')][3][-1]
            time_serr = full_results[(ml_alg_ind,\
                                setting, dacs_alpha_ind, 'dacs')][4][-1]
            timeR.append(time_avg)
            timeLabelR.append(" ")
            methodR.append("DACS")
            serrR.append(time_serr)
            settingR.append(settings[setting])
            ml_algR.append(ml_algs[ml_alg_ind])
            alphaR.append(dacs_alpha_ind)


time_dfR = pd.DataFrame({
    'Time (sec)': timeR,
    'time': timeLabelR,
    'serr': serrR,
    'ML Alg': ml_algR,
    'setting': settingR,
    'alpha_var': alphaR
})

time_dfR.to_csv('./csvs_to_plot_100/time_R.csv')
###################################################################################################
pi0R = []
sim_settingR = []
labelR = []
serrR = []
alpha_ind = 0
for ml_alg_ind in range(3):
    for setting in range(6):
        [pi0_avg, pi0_serr] = full_results[(ml_alg_ind, setting, alpha_ind, 'env')][:2]
        pi0R.append(pi0_avg)
        sim_settingR.append(settings[setting])
        labelR.append('pi0')
        serrR.append(pi0_serr)

pi0_dfR = pd.DataFrame({
    'pi0': pi0R,
    'sim setting': sim_settingR,
    'pi0_x': labelR,
    'serr': serrR
})

pi0_dfR.to_csv('./csvs_to_plot_100/pi0_results_R.csv')
###################################################################################################
pi0cR = []
clusterR = []
sim_settingR = []
serrR = []
alpha_ind = 0
for ml_alg_ind in range(3):
    for setting in range(6):
        [pi0c_avg, pi0c_serr] = full_results[(ml_alg_ind, setting, alpha_ind, 'env')][2:]
        for c in range(len(pi0c_avg)):
            pi0cR.append(pi0c_avg[c])
            sim_settingR.append(settings[setting])
            serrR.append(pi0c_serr[c])
            clusterR.append(c)

pi0c_dfR = pd.DataFrame({
    'pi0 per cluster': pi0cR,
    'sim setting': sim_settingR,
    'serr': serrR,
    'cluster': clusterR
})

pi0c_dfR.to_csv('./csvs_to_plot_100/pi0c_results_R.csv')