import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

print("Starting")


alphas = ['alpha=0.05', 'alpha=0.2', 'alpha=0.35']


time_dict = dict()
ppf_dict = dict()
st_dict = dict()

ppf_pgd_mosek_dict = dict()
pgd_time_dict = dict()

for (warm_or_custom, couple) in [(True, True), (False, True), (True, False), (False, False)]:
    for alpha_ind in range(3):
        for setting in range(2):
            time_dict[warm_or_custom,couple,alpha_ind,setting] = [[],[]]
            ppf_dict[warm_or_custom,couple,alpha_ind,setting] = []
            st_dict[warm_or_custom,couple,alpha_ind,setting] = []
            ppf_pgd_mosek_dict[warm_or_custom,couple,alpha_ind,setting] = []
            pgd_time_dict[warm_or_custom,couple,alpha_ind,setting] = []

for (warm_or_custom, couple) in [(True, True), (False, True), (True, False)]:
    ppf_df_array_dacs_val = []
    ppf_df_array_jc_val = []
    ppf_df_array_setting = []
    ppf_df_array_alpha = []


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
    for alpha_ind in range(3):
        for setting in [0,1]:
            results = \
                pd.read_csv(f'collated_sharpe_results/metrics_s{setting}_a{alpha_ind}_w{warm_or_custom}_c{couple}.csv', header=None).to_numpy().astype('float')
            
            vanilla_results = \
                pd.read_csv(f'collated_sharpe_results/vanilla_metrics_s{setting}_a{alpha_ind}.csv', header=None).to_numpy().astype('float')
            
            time_results = \
                pd.read_csv(f'collated_sharpe_results/solver_times_s{setting}_a{alpha_ind}_w{warm_or_custom}_c{couple}.csv', header=None).to_numpy().astype('float')

            
            print("Loading histogram results...")
            histogram_results = []

            with open(f'collated_sharpe_results/histogram_s{setting}_a{alpha_ind}.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    histogram_results.append([float(num) for num in row])



            ppfs = []
            vanilla_ppfs = []
            data = []

            mt_metric_dacs_fdr_arr = []
            mt_metric_vanilla_fdr_arr = []

            mt_metric_dacs_power_arr = []
            mt_metric_vanilla_power_arr = []

            mt_metric_dacs_numr_arr = []
            mt_metric_vanilla_numr_arr = []

            time_arr = []
            for job in tqdm(range(250)):
                time_dict[warm_or_custom,couple,alpha_ind,setting][0].append(time_results[job][0])
                time_dict[warm_or_custom,couple,alpha_ind,setting][1].append(time_results[job][1])

                ppf_df_array_alpha.append(alphas[alpha_ind])
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
                

                time_arr.append(result[-1])


                histogram_result = histogram_results[job]

                ppf = np.sum(histogram_result <= diversity_result)/len(histogram_result)
                ppf_df_array_dacs_val.append(ppf)

                ppf_dict[warm_or_custom,couple,alpha_ind,setting].append(ppf)
                

                block_indexer = results[job][5]
                st_dict[warm_or_custom,couple,alpha_ind,setting].append(block_indexer)


                ppf_df_array_setting.append(['Setting 1', 'Setting 2'][setting])

                vanilla_ppf = np.sum(histogram_result <= vanilla_diversity_result)/len(histogram_result)
                ppf_df_array_jc_val.append(vanilla_ppf)


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

    sharpe_ppf_results.to_csv(f'./sharpe_csvs_to_plot/sharpe_ppf_results_w{warm_or_custom}_c{couple}.csv')


    sharpe_mt_metric_results = pd.DataFrame({
        'method': mt_metric_method,
        'Setting': mt_metric_setting,
        'serr': mt_metric_serr,
        'alpha': mt_metric_alpha,
        'error_metric': mt_metric_fdr_power_label,
        'rate': mt_metric_rate
    })

    sharpe_mt_metric_results.to_csv(f'./sharpe_csvs_to_plot/sharpe_mt_metric_results_w{warm_or_custom}_c{couple}.csv')


    sharpe_mt_numr_results = pd.DataFrame({
        'method': mt_metric_numr_method,
        'Setting': mt_metric_numr_setting,
        'serr': mt_metric_numr_serr,
        'alpha': mt_metric_numr_alpha,
        'num_rejections': mt_metric_numr_label,
        '#R': mt_metric_numr_rate
    })

    sharpe_mt_numr_results.to_csv(f'./sharpe_csvs_to_plot/sharpe_mt_numr_results_w{warm_or_custom}_c{couple}.csv')



ppf_df_array_dacs_val = []
ppf_df_array_jc_val = []
ppf_df_array_setting = []
ppf_df_array_alpha = []
for alpha_ind in range(3):
    for setting in [0,1]:
        results = \
            pd.read_csv(f'collated_sharpe_results/metrics_s{setting}_a{alpha_ind}_w{False}_c{False}.csv', header=None).to_numpy().astype('float')
        
        vanilla_results = \
                pd.read_csv(f'collated_sharpe_results/vanilla_metrics_s{setting}_a{alpha_ind}.csv', header=None).to_numpy().astype('float')

        print("Loading histogram results...")
        histogram_results = []

        with open(f'collated_sharpe_results/histogram_s{setting}_a{alpha_ind}.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                histogram_results.append([float(num) for num in row])

        time_arr = []
        ppfs = []
        vanilla_ppfs = []
        data = []
        for job in tqdm(range(250)):
            time_dict[False,False,alpha_ind,setting][0].append(results[job][2])
            time_dict[False,False,alpha_ind,setting][1].append(results[job][1])

            result = results[job]
            vanilla_result = vanilla_results[job]


            diversity_result = result[0]
            vanilla_diversity_result = vanilla_result[-1]


            histogram_result = histogram_results[job]

            ppf = np.sum(histogram_result <= diversity_result)/len(histogram_result)
            ppf_df_array_dacs_val.append(ppf)
            ppf_dict[False,False,alpha_ind,setting].append(ppf)


            block_indexer = result[3]
            st_dict[False,False,alpha_ind,setting].append(block_indexer)


            ppf_df_array_setting.append(['Setting 1', 'Setting 2'][setting])

            vanilla_ppf = np.sum(histogram_result <= vanilla_diversity_result)/len(histogram_result)
            ppf_df_array_jc_val.append(vanilla_ppf)
            ppf_df_array_alpha.append(alphas[alpha_ind])


sharpe_ppf_results = pd.DataFrame({
        'DACS': ppf_df_array_dacs_val,
        'Setting': ppf_df_array_setting,
        'CS': ppf_df_array_jc_val,
        'alpha': ppf_df_array_alpha
    })

sharpe_ppf_results.to_csv(f'./sharpe_csvs_to_plot/sharpe_ppf_results_w{False}_c{False}.csv')





#### comparison results
ppf_df_array_setting = []
ppf_df_array_alpha = []
ppf_true_true = []
ppf_true_false = []
for setting in range(2):
    for alpha_ind in range(3):
        for __ in range(len(ppf_dict[True,True,alpha_ind,setting])):
            ppf_df_array_setting.append(['Setting 1', 'Setting 2'][setting])
            ppf_df_array_alpha.append(alphas[alpha_ind])
        ppf_true_true = ppf_true_true + ppf_dict[True,True,alpha_ind,setting]
        ppf_true_false = ppf_true_false + ppf_dict[True,False,alpha_ind,setting]


sharpe_ppf_results_comp1 = pd.DataFrame({
        'Warm-started PGD with coupling': ppf_true_true,
        'Setting': ppf_df_array_setting,
        'PGD without coupling or warm-starting': ppf_true_false,
        'alpha': ppf_df_array_alpha
    })

sharpe_ppf_results_comp1.to_csv(f'./sharpe_csvs_to_plot/sharpe_ppf_results_comp1.csv')



for setting in range(2):
    for alpha_ind in range(3):
        print(f'Setting {setting}, Alpha {alpha_ind}, Time: {np.mean(pgd_time_dict[True,False,alpha_ind,setting])}')


#### comparison results
ppf_df_array_setting = []
ppf_df_array_alpha = []
ppf_false_false = []
ppf_true_false = []
for setting in range(2):
    for alpha_ind in range(3):
        for __ in range(len(ppf_dict[False,False,alpha_ind,setting])):
            ppf_df_array_setting.append(['Setting 1', 'Setting 2'][setting])
            ppf_df_array_alpha.append(alphas[alpha_ind])
        ppf_false_false = ppf_false_false + ppf_dict[False,False,alpha_ind,setting]
        ppf_true_false = ppf_true_false + ppf_dict[True,False,alpha_ind,setting]


sharpe_ppf_results_comp2 = pd.DataFrame({
        'MOSEK (no coupling)': ppf_false_false,
        'Setting': ppf_df_array_setting,
        'PGD without coupling or warm-starting': ppf_true_false,
        'alpha': ppf_df_array_alpha
    })

sharpe_ppf_results_comp2.to_csv(f'./sharpe_csvs_to_plot/sharpe_ppf_results_comp2.csv')





#### comparison results
block_ind_df_array_setting = []
block_ind_array_alpha = []
block_ind_false_false = []
block_ind_true_false = []
for setting in range(2):
    for alpha_ind in range(3):
        for __ in range(len(ppf_dict[False,False,alpha_ind,setting])):
            block_ind_df_array_setting.append(['Setting 1', 'Setting 2'][setting])
            block_ind_array_alpha.append(alphas[alpha_ind])
        block_ind_false_false = block_ind_false_false + st_dict[False,False,alpha_ind,setting]
        block_ind_true_false = block_ind_true_false + st_dict[True,False,alpha_ind,setting]




true_true_times = []
false_true_times = []
true_false_times = []
false_false_times = []
settingR = []
alphaR = []

for alpha_ind in range(3):
    for setting in range(2):
        for job_ind in range(250):
            true_true_times.append(time_dict[True, True, alpha_ind, setting][0][job_ind])
            true_false_times.append(time_dict[True, False, alpha_ind, setting][0][job_ind])
            false_true_times.append(time_dict[False, True, alpha_ind, setting][0][job_ind])
            false_false_times.append(time_dict[False, False, alpha_ind, setting][0][job_ind])
            settingR.append(['Setting 1', 'Setting 2'][setting])
            alphaR.append(alphas[alpha_ind])

sharpe_st_results_comp = pd.DataFrame({
        'MOSEK (no coupling)': block_ind_true_false,
        'Setting': block_ind_df_array_setting,
        'PGD without coupling or warm-starting': block_ind_true_false,
        'alpha': block_ind_array_alpha
    })

sharpe_st_results_comp.to_csv(f'./sharpe_csvs_to_plot/sharpe_st_results_comp.csv')


def latex_fmt(mean, std, n):
    err = 1.96 * std / np.sqrt(n)
    return f"${mean:.2f} \\pm {err:.2f}$"

method_labels = {
    (True, True): "Warm-started PGD with coupling",
    (False, True): "PGD with coupling only",
    (True, False): "PGD without coupling or warm-starting",
    (False, False): "MOSEK (no coupling)"
}

alpha_vals = ["0.05", "0.2", "0.35"]

def get_latex_entry(arr):
    return latex_fmt(np.mean(arr), np.std(arr), len(arr))

def generate_custom_latex_table(df, setting):
    """Generate LaTeX table with custom formatting including multirow"""
    
    # Sort the dataframe
    method_order = [
        "Warm-started PGD with coupling",
        "PGD with coupling only", 
        "PGD without coupling or warm-starting",
        "MOSEK (no coupling)"
    ]
    df["Method"] = pd.Categorical(df["Method"], categories=method_order, ordered=True)
    df = df.sort_values(by=["$\\alpha$", "Method"])
    
    # Start building the LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{Solver and total times for Sharpe ratio in Setting {setting + 1}}}")
    latex_lines.append(f"\\label{{tab:sharpe_times_setting{setting + 1}}}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("$\\alpha$ & Method & Solver times & Total times \\\\")
    latex_lines.append("\\midrule")
    
    # Group by alpha values
    alpha_groups = df.groupby("$\\alpha$")
    
    for i, (alpha, group) in enumerate(alpha_groups):
        # Add horizontal line between alpha groups (except before first)
        if i > 0:
            latex_lines.append("\\hline")
        
        # Process each method for this alpha
        for j, (_, row) in enumerate(group.iterrows()):
            if j == 0:
                # First row for this alpha - include multirow
                alpha_cell = f"\\multirow{{4}}{{*}}{{{alpha}}}"
            else:
                # Subsequent rows - empty alpha cell
                alpha_cell = ""
            
            # Format the row
            method = row["Method"]
            solver_times = row["Solver times"]
            total_times = row["Total times"]
            
            latex_lines.append(f"{alpha_cell} & {method} & {solver_times} & {total_times} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)

# Main loop to generate tables for each setting
for setting in range(2):
    ratios = []
    reductions = []
    rows = []
    for alpha_ind, alpha_str in enumerate(alpha_vals):
        for warm, couple in method_labels:
            solver_times = time_dict[warm, couple, alpha_ind, setting][0]
            total_times = time_dict[warm, couple, alpha_ind, setting][1]
            rows.append({
                "$\\alpha$": f"${alpha_str}$",  # use LaTeX alpha symbol
                "Method": method_labels[(warm, couple)],
                "Solver times": get_latex_entry(solver_times),
                "Total times": get_latex_entry(total_times)
            })

        ratios.append(np.mean(time_dict[False, False, alpha_ind, setting][1])/np.mean(time_dict[True, False, alpha_ind, setting][1]))
        reductions.append(1 - np.mean(time_dict[True, True, alpha_ind, setting][1])/np.mean(time_dict[True, False, alpha_ind, setting][1]))


    print(f'Intervals for Setting {setting+1}: [{np.min(ratios), np.max(ratios)}], [{np.min(reductions), np.max(reductions)}]')
    
    df_combined = pd.DataFrame(rows)
    
    # Generate custom LaTeX table
    latex_table = generate_custom_latex_table(df_combined, setting)
    
    # Write to file
    with open(f"sharpe_tables/sharpe_combined_setting{setting}.tex", "w") as f:
        f.write(latex_table)
