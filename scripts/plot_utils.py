import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from neuralforecast.losses.numpy import mae, rmse


def _evaluate_ohiot1dm(Y_hat_df, av_mask, model_name, metric, values):
    results_df = Y_hat_df.copy()
    # Filter values with at least 1 available mask in input window
    results_df = results_df.merge(av_mask[['unique_id', 'cutoff', 'sum_av_mask']], on=['unique_id', 'cutoff'], how='left')
    results_df = results_df[results_df['sum_av_mask'] > 0].reset_index(drop=True)
    # Filter ffill values of y
    results_df = results_df[results_df.available_mask==1]
    # Keep critical values of y

    if values == 'critical':
        results_df = results_df[(results_df.y<=70) | (results_df.y>=180)]

    if metric == 'mae':
        metric_result = mae(results_df['y'], results_df[model_name])
    elif metric == 'rmse':
        metric_result = rmse(results_df['y'], results_df[model_name])
    
    return metric_result

def _evaluate_simglucose(Y_hat_df, model_name, metric, values):
    results_df = Y_hat_df.copy()

    if values == 'critical':
        results_df = results_df[(results_df.y<=70) | (results_df.y>=180)]

    if metric == 'mae':
        metric_result = mae(results_df['y'], results_df[model_name])
    elif metric == 'rmse':
        metric_result = rmse(results_df['y'], results_df[model_name])
    
    return metric_result

def get_global_model_results(df_path, model_name, pat_ids, dataset_name, trials, exp_name, metric, values):
    results = []
    for pat_id in pat_ids:
        result = []
        for trial in range(trials):
            Y_hat_df = pd.read_csv(f'{df_path}/{dataset_name}_exog_6/{exp_name}_models/trial_{exp_name}_{trial}/forecasts.csv')
            Y_hat_df = Y_hat_df[Y_hat_df.unique_id==pat_id]

            if dataset_name=='ohiot1dm':
                av_mask = pd.read_csv('../datasets/ohiot1dm_exog_9_day_test_avmask.csv')
                m = _evaluate_ohiot1dm(Y_hat_df, av_mask, model_name, metric, values)
            elif dataset_name=='simglucose':
                m = _evaluate_simglucose(Y_hat_df, model_name, metric, values)
    
            result.append(m) 
        results.append(result)
    results = np.array(results).T
    print('done')

    return results

def get_local_model_results(df_path, model_name, pat_ids, dataset_name, trials, exp_name, metric, values):
    results = []
    for pat_id in pat_ids:
        result = []
        for trial in range(trials):
            Y_hat_df = pd.read_csv(f'{df_path}/{dataset_name}_exog_{pat_id}_6/{exp_name}_models/trial_{exp_name}_{trial}/forecasts.csv')
            Y_hat_df = Y_hat_df[Y_hat_df.unique_id==pat_id]
            
            if dataset_name=='ohiot1dm':
                av_mask = pd.read_csv('../datasets/ohiot1dm_exog_9_day_test_avmask.csv')
                m = _evaluate_ohiot1dm(Y_hat_df, av_mask, model_name, metric, values)
            elif dataset_name=='simglucose':
                m = _evaluate_simglucose(Y_hat_df, model_name, metric, values)

            result.append(m) 
        results.append(result)
    results = np.array(results).T
    print('done')

    return results

def window_tpr(df, event, model_name, threshold):
    '''
    Calculates the window-level true positive rate (TPR) for a model predicting 
    critical glycemic events (hypoglycemia or hyperglycemia). A "true positive window" 
    is defined as a forecast window where the event is present in the ground truth, and 
    the model prediction also indicates the event based on the specified threshold.

    Inputs:
    ---------
        df: Dataframe output from NeuralForecast model training, containing both model predictions and ground truth values.
        event: String indicating the type of critical event to evaluate; either 'hypoglycemia' or 'hyperglycemia'.
        model_name: Name of the column in the dataframe that contains the model’s predictions.
        threshold: Integer value used to define the threshold for the critical event.

    Output:
    ---------
        True positive rate (TPR), calculated at the forecast window level.
    '''
    
    # Find forecast windows with critical event based on the ground truth values
    if event=='hypoglycemia':
        check = df.groupby('cutoff').y.min()
        event_windows = check[check<=threshold]
    elif event=='hyperglycemia':
        check = df.groupby('cutoff').y.max()
        event_windows = check[check>=threshold]

    # Take subset of forecast windows with critical events
    subset = df.set_index('cutoff').loc[event_windows.index.unique()]

    # Determine if model prediction meets threshold for critical event.
    if event=='hypoglycemia':
        subset_confirm = subset.groupby('cutoff')[model_name].min()
        subset_confirm_count = subset_confirm[subset_confirm<=threshold].shape[0]
    elif event=='hyperglycemia':
        subset_confirm = subset.groupby('cutoff')[model_name].max()
        subset_confirm_count = subset_confirm[subset_confirm>=threshold].shape[0]

    return subset_confirm_count/event_windows.shape[0]


def window_fnr(df, event, model_name, threshold): # would be FNR = 1 - TPR
    '''
    Calculates the window-level false negative rate (FNR) for a model predicting 
    critical glycemic events (hypoglycemia or hyperglycemia). A "false negative rate" 
    is defined as a forecast window where the event is present in the ground truth, but 
    the model prediction indicates no event based on the specified threshold.

    Inputs:
    ---------
        df: Dataframe output from NeuralForecast model training, containing both model predictions and ground truth values.
        event: String indicating the type of critical event to evaluate; either 'hypoglycemia' or 'hyperglycemia'.
        model_name: Name of the column in the dataframe that contains the model’s predictions.
        threshold: Integer value used to define the threshold for the critical event.

    Output:
    ---------
        False negative rate (FNR), calculated at the forecast window level.
    '''

    # Find forecast windows with critical event based on the ground truth values
    if event=='hypoglycemia':
        check = df.groupby('cutoff').y.min()
        event_windows = check[check<=threshold] 
    elif event=='hyperglycemia':
        check = df.groupby('cutoff').y.max()
        event_windows = check[check>=threshold]

    # Take subset of forecast windows with critical events
    subset = df.set_index('cutoff').loc[event_windows.index.unique()]

    # Determine if model prediction meets threshold for critical event.
    if event=='hypoglycemia':
        subset_confirm = subset.groupby('cutoff')[model_name].min()
        subset_confirm_count = subset_confirm[subset_confirm>=threshold].shape[0] # flip sign compared to TPR
    elif event=='hyperglycemia':
        subset_confirm = subset.groupby('cutoff')[model_name].max()
        subset_confirm_count = subset_confirm[subset_confirm<=threshold].shape[0] # flip sign compared to TPR

    return subset_confirm_count/event_windows.shape[0]


def window_fpr(df, event, model_name, threshold):
    '''
    Calculates the window-level false positive rate (FPR) for a model predicting 
    critical glycemic events (hypoglycemia or hyperglycemia). A "false positive rate" 
    is defined as a forecast window where no event is present in the ground truth, but 
    the model prediction indicates an event based on the specified threshold.

    Inputs:
    ---------
        df: Dataframe output from NeuralForecast model training, containing both model predictions and ground truth values.
        event: String indicating the type of critical event to evaluate; either 'hypoglycemia' or 'hyperglycemia'.
        model_name: Name of the column in the dataframe that contains the model’s predictions.
        threshold: Integer value used to define the threshold for the critical event.

    Output:
    ---------
        False positive rate (FPR), calculated at the forecast window level.
    '''

    # Find forecast windows without critical events based on the ground truth values
    if event=='hypoglycemia':
        check = df.groupby('cutoff').y.min()
        event_windows = check[check>=threshold] # flip sign compared to TPR
    elif event=='hyperglycemia':
        check = df.groupby('cutoff').y.max()
        event_windows = check[check<=threshold] # flip sign compared to TPR

    # Take subset of forecast windows with critical events
    subset = df.set_index('cutoff').loc[event_windows.index.unique()]

    # Determine if model prediction meets threshold for critical event.
    if event=='hypoglycemia':
        subset_confirm = subset.groupby('cutoff')[model_name].min()
        subset_confirm_count = subset_confirm[subset_confirm<=threshold].shape[0]
    elif event=='hyperglycemia':
        subset_confirm = subset.groupby('cutoff')[model_name].max()
        subset_confirm_count = subset_confirm[subset_confirm>=threshold].shape[0] 

    return subset_confirm_count/event_windows.shape[0]



def patient_glucose_examples_plot(data, pat_list, color_palette, xlim, ylim, insulin_scale, save_dir=None):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'serif' 

    # add lines for meal and bolus
    fig, axs = plt.subplots(nrows=6, figsize=(20, 21),
                            sharex=True, sharey=True,
                            layout='constrained')

    for ax, pat in zip(axs, pat_list):
        pat_data = data[data.unique_id==pat]

        time_hour = (pat_data.set_index('ds').index - pat_data.ds.iloc[0]) / pd.Timedelta('1D')

        ax.plot(time_hour, pat_data.y, color='black', linewidth=1.5)
        ax.set_title(pat.capitalize(), size=26)
        ax.tick_params(axis="y", labelsize=26, color='black')
        ax.tick_params(axis="x", labelsize=26)

        # Create a twin Axes for the right y-axis
        ax_right = ax.twinx()
        ax_right.set_ylim([0, ylim])
        ax_right.set_yticklabels([f'{i:.2f}' for i in np.round(np.linspace(0, 400/insulin_scale, 3), 1)])
        ax_right.tick_params(axis="y", labelsize=26, colors='black')
        ax_right.grid(False)
        ax_right.plot(time_hour, pat_data.bolus_insulin*insulin_scale, color=color_palette[10], linewidth=1.5, label='Bolus Insulin')
        ax_right.plot(time_hour, pat_data.basal_insulin*insulin_scale, color=color_palette[2], linewidth=1.5, label='Basal Insulin')

    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    axs[-1].set_xlabel('Time (Days)', fontsize=26)
    fig.supylabel('Blood Glucose (mg/dL)', fontsize=26, color='black')
    
    # Add super ylabel on the left side
    fig.text(1, 0.5, 'Insulin (U)', va='center', rotation='vertical', color='black', fontsize=26)
    
    plt.legend(bbox_to_anchor =(1.05,-0.45), loc='lower right', ncol=2, fontsize=26, markerscale=0.5)
    
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')



def insulin_kde_plots(df, ids, color_palette, xlim, save_dir=None):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'serif'

    fig, axes = plt.subplots(2, 6, figsize=(20, 10))
    axes = axes.flatten()

    # Loop over the axes and plot KDE for each column of data
    for (ax, id_) in zip(axes, ids):
        basal = df[df.unique_id==id_].basal_insulin.values
        basal = basal[basal!=0]
        bolus = df[df.unique_id==id_].bolus_insulin.values
        bolus = bolus[bolus!=0]
        
        bin_width=1
        basal_bins = np.arange(-2, 20, bin_width)
        bolus_bins = np.arange(-2, 20, bin_width)
        ax.hist(bolus, bins=bolus_bins, color=color_palette[10], alpha=0.6, density=True, label='Bolus Insulin')
        ax.hist(basal, bins=basal_bins, color=color_palette[2], alpha=0.6, density=True, label='Basal Insulin')
        ax.set_xlim([-2, xlim])
        ax.set_ylim([0, 1.05])
        ax.set_title(f'{id_}', fontsize=22)
        ax.tick_params(axis="y", labelsize=22)
        ax.tick_params(axis="x", labelsize=22)

    fig.supylabel('Density', fontsize=26)
    fig.supxlabel('Insulin Value', fontsize=26)
    fig.tight_layout()
    plt.legend(bbox_to_anchor =(1.05,-0.30), loc='lower right', ncol=2, fontsize=26, markerscale=0.5)

    if save_dir is not None:
        plt.savefig(save_dir)



def patient_summary_boxplot(group1, group2, label1, label2, ylim, save_dir):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    COLORS = ['#78ACA8', '#CA6F6A', '#7B3841', '#D5BC67', '#8b8b8b', 
          '#235796', '#E77A5B', '#628793']

    boxprops1 = dict(color='black', linewidth=2, linestyle='--')
    boxprops2 = dict(color='black', linewidth=2, linestyle='-')
    medianprops1 = dict(linestyle='-', linewidth=2,)
    medianprops2 = dict(linestyle='--', linewidth=2,)
    whiskerprops1 = dict(linestyle='-',linewidth=2,)
    whiskerprops2 = dict(linestyle='-',linewidth=2,)
    capprops1 = dict(color='black', linewidth=2)  # Increased cap size
    capprops2 = dict(color='black', linewidth=2)  # In

    data_a = group1
    data_b = group2

    positions_a = [2]
    positions_b = [0]

    ticks = ['Local', 'Global-Local']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['caps'], color='black')
        plt.setp(bp['medians'], color='black')

    plt.figure(figsize=(9, 6))

    bpl = plt.boxplot(data_a, positions=positions_a, sym='', 
                      boxprops=boxprops1, 
                      medianprops=medianprops1,
                      whiskerprops=whiskerprops1,
                      capprops=capprops1,
                      patch_artist=True, 
                      widths=0.6)
    bpr = plt.boxplot(data_b, positions=positions_b, sym='', 
                      boxprops=boxprops2, 
                      medianprops=medianprops2,
                      whiskerprops=whiskerprops2,
                      capprops=capprops2,
                      patch_artist=True, 
                      widths=0.6)
    set_box_color(bpl, COLORS[6]) # colors are from http://colorbrewer2.org/
    set_box_color(bpr, COLORS[0])

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=COLORS[6], label=label1, linewidth=15)
    plt.plot([], c=COLORS[0], label=label2, linewidth=15)

    #plt.legend(fontsize=28, loc='upper right')
    plt.ylabel('MAE', fontsize=31)
    plt.xlabel('NHITS-PK Model', fontsize=31)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir)



def patient_id_boxplot(group1, group2, label1, label2, ylim, save_dir):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    COLORS = ['#78ACA8', '#CA6F6A', '#7B3841', '#D5BC67', '#8b8b8b', 
          '#235796', '#E77A5B', '#628793']

    boxprops1 = dict(color='black', linewidth=3, linestyle='-')
    boxprops2 = dict(color='black', linewidth=3, linestyle='-')
    medianprops1 = dict(linestyle='-', linewidth=3,)
    medianprops2 = dict(linestyle='--', linewidth=3,)
    whiskerprops1 = dict(linestyle='-',linewidth=3,)
    whiskerprops2 = dict(linestyle='-',linewidth=3,)
    capprops1 = dict(color='black', linewidth=3)  # Increased cap size
    capprops2 = dict(color='black', linewidth=3)  # In

    data_a = group1
    data_b = group2

    positions_a = [i*2+0.4 for i in list(range(12))]
    positions_b = [i*2-0.4 for i in list(range(12))]

    ticks = ['540', '544', '552', '559', '563', '567', '570',
             '575', '584', '588', '591', '596']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['caps'], color='black')
        plt.setp(bp['medians'], color='black')

    plt.figure(figsize=(25, 17))

    bpl = plt.boxplot(data_a, positions=positions_a, sym='', 
                      boxprops=boxprops1, 
                      medianprops=medianprops1,
                      whiskerprops=whiskerprops1,
                      capprops=capprops1,
                      patch_artist=True, 
                      widths=0.6)
    bpr = plt.boxplot(data_b, positions=positions_b, sym='', 
                      boxprops=boxprops2, 
                      medianprops=medianprops2,
                      whiskerprops=whiskerprops2,
                      capprops=capprops2,
                      patch_artist=True, 
                      widths=0.6)
    set_box_color(bpl, COLORS[6]) # colors are from http://colorbrewer2.org/
    set_box_color(bpr, COLORS[0])

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=COLORS[0], linestyle='-', label=label2, linewidth=15)
    plt.plot([], c=COLORS[6], label=label1, linewidth=15)

    plt.legend(fontsize=60, loc='lower right', frameon=True, fancybox=True)
    plt.xlabel('Unique Patient ID', fontsize=70)
    plt.ylabel('MAE', fontsize=70)
    plt.xticks(fontsize=56)
    plt.yticks(fontsize=56)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir)



def time_gain_plot(ids, values, save_dir=None):
    # Sample data
    info = {
        'Category': [f"{i.replace('#', '')}" for i in ids], 
        'Values': values
           }
    df = pd.DataFrame(info)
    
    # Determine colors based on positive or negative values
    colors = ['green' if val >= 0 else 'red' for val in df['Values']]
    
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(25, 12))
    bars = plt.bar(df['Category'], df['Values'], color=colors, alpha=0.6)
    
    # Add labels to the top of the bars
    for bar in bars:
        yval = bar.get_height()
        offset = 0.1 if yval >= 1 else -0.01  # Offset value
        plt.text(bar.get_x() + bar.get_width() / 2, yval+offset, round(yval, 2), ha='center', va='bottom' if yval >= 0 else 'top', fontsize=48)
    
    plt.ylim([-4, 4.3])
    plt.xlabel('Unique Patient ID', fontsize=70)
    plt.ylabel('Time (Minutes)', fontsize=70)
    plt.xticks(fontsize=56)
    plt.yticks(fontsize=56)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir)


