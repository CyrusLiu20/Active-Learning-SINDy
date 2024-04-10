import torch
import numpy as np
import matplotlib.pyplot as plt

folder_name = "SimulationData/"
# folder_name = ""

U_history_all = torch.load(folder_name+"U_history_all.pt")
X_history_all = torch.load(folder_name+"X_history_all.pt")
Hyperparams_history_all = torch.load(folder_name+"Hyperparams_history_all.pt")

Hyperparams_random_all, Hyperparams_optimal_all = [], []
for Hyperparams_history in Hyperparams_history_all:
    Hyperparams_random_all.append(np.array(Hyperparams_history[0]).T)
    Hyperparams_optimal_all.append(np.array(Hyperparams_history[1]).T)

error_history_all = torch.load(folder_name+"error_history_all.pt")
error_history_all = [tensor for error_history in error_history_all for tensor in error_history]
epochs = torch.load(folder_name+"epochs.pt")
metrics = torch.load(folder_name+"metrics.pt")

n_experiments = int(len(error_history_all)/2)
n_epochs = np.sum(np.array(epochs))
n_metrics = len(metrics)

# Figure 1
xlabel = torch.cumsum(epochs.clone().detach(),0).clone().detach()
for m in range(n_metrics):
    plt.figure(m+101)
    for i in range(len(error_history_all)):
        if np.mod(i,2) == 0:
            plt.plot(xlabel,torch.tensor(error_history_all[i])[:,m],'-o',color = "blue")
        else:
            plt.plot(xlabel,torch.tensor(error_history_all[i])[:,m],'-o',color = "orange")

    plt.ticklabel_format(useOffset=False, style='plain')
    plt.xlabel("Epochs")
    plt.ylabel("error")
    plt.yscale('log')
    plt.title(f"All Exploration - Error Plot {metrics[m]}")
    plt.legend(["Random Exploration","Trace Exploration"])

    figure_name = folder_name + f"pics\\All {str(n_experiments)} Experiments_{metrics[m]}"
    plt.savefig(figure_name,dpi=300)

# # Figure 2
for m in range(n_metrics):
    plt.figure(m+201)
    n_cols = 5
    fig, axes = plt.subplots(nrows=int(n_experiments/n_cols), ncols=n_cols, figsize=(15, 6))
    # Flatten the 2D array of subplots into a 1D array
    axes = axes.flatten()

    # Plot data on each subplot
    for i, ax in enumerate(axes):
        ax.plot(xlabel, torch.tensor(error_history_all[2*i])[:,m],'-o',color='blue' )
        ax.plot(xlabel, torch.tensor(error_history_all[2*i+1])[:,m],'-o',color='orange')
        ax.set_yscale('log')
        ax.set_title(f'Experiment {i+1}')

        
    figure_name = folder_name + f"pics\\All {n_experiments} Experiments_{metrics[m]} (Detailed)"
    plt.legend(["Random Exploration","Trace Exploration"])
    plt.savefig(figure_name,dpi=300)


# Figure 3
error_tensor = torch.tensor(error_history_all)
random_exploration, trace_exploration = error_tensor[::2], error_tensor[1::2]
random_exploration[random_exploration.isnan()] = 1e20
trace_exploration[trace_exploration.isnan()] = 1e20
random_median = torch.median(random_exploration,0).values
trace_median = torch.median(trace_exploration,0).values
for m in range(n_metrics):
    plt.figure(m+301)
    plt.plot(xlabel,random_median[:,m],'-o',color='blue')
    plt.plot(xlabel,trace_median[:,m],'-o',color='orange')
    plt.legend(["Random Exploration","Trace Exploration"])
    plt.title("Median of model loss")
    plt.yscale('log')
    figure_name = folder_name + f"pics\\Median of model loss_{metrics[m]}"
    plt.savefig(figure_name,dpi=300)


# Figure 4
for m in range(n_metrics):
    plt.figure(m+401)
    # Number of times Optimal Exploration wins
    count_rand, count_optimal = 0, 0
    for i in range(0, n_experiments*2, 2):
        error_rand = torch.tensor(error_history_all[i])[-1,m]
        error_optimal = torch.tensor(error_history_all[i+1])[-1,m]

        error_rand = error_rand if not np.isnan(error_rand) else 1e20
        error_optimal = error_optimal if not np.isnan(error_optimal) else 1e20

        count_optimal += 1 if error_rand > error_optimal else 0
        count_rand += 1 if error_optimal > error_rand else 0
            
    labels = ["Random Exploration", "Trace Exploration", "Indeterminate"]
    data = [count_rand, count_optimal, n_experiments-count_rand-count_optimal]
    plt.pie(data, labels=labels, autopct=lambda p: '{:.0f}'.format(p * sum(data) / 100), startangle=90)
    plt.title(f"Number of Times Each Exploration Won_{metrics[m]}")

    figure_name = folder_name + f"pics\\Number of Times Each Exploration Won_{metrics[m]}"
    plt.savefig(figure_name,dpi=300)


# Parameter analysis
def plot_single_yaxis(x, y1, line1):
    ax1.plot(x, y1, line1)
def plot_dual_yaxis(x, y1, y2, line1, line2):
    ax1.plot(x, y1, line1)
    ax2.plot(x, y2, line2)

fig = plt.figure(num=5)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
epochs_range = range(1,int(n_epochs+1))
rand_plot, optimal_plot = False, True
n_params = Hyperparams_random_all[0].shape[0]
for i in range(len(Hyperparams_history_all)):
    if rand_plot:
        if n_params == 1:
            plot_single_yaxis(epochs_range, Hyperparams_random_all, "b-")
        elif n_params ==2:
            plot_dual_yaxis(epochs_range, Hyperparams_random_all[i][0], Hyperparams_random_all[i][1], "b-", "g-")
    if optimal_plot:
        if n_params == 1:
            plot_single_yaxis(epochs_range, Hyperparams_random_all, "b-")
        elif n_params ==2:
            plot_dual_yaxis(epochs_range, Hyperparams_optimal_all[i][0], Hyperparams_optimal_all[i][1], "y-", "r-")
plt.axvline(x=epochs[0], color='black', linestyle='--', label='Vertical Line')
plt.title("Hyperparameter analysis")
plt.ylabel("Hyperparameter")

figure_name = folder_name+"pics\\Hyperparameters analysis"
plt.savefig(figure_name,dpi=300)


for m in range(n_metrics):
    plt.figure(m+601)

    plt.plot(xlabel,random_median[:,m],'-o',color='blue')
    plt.plot(xlabel,trace_median[:,m],'-o',color='orange')
    for i in range(len(error_history_all)):
        if np.mod(i,2) == 0:
            plt.plot(xlabel,torch.tensor(error_history_all[i])[:,m],'-o',color = "blue", alpha=0.1)
        else:
            plt.plot(xlabel,torch.tensor(error_history_all[i])[:,m],'-o',color = "orange", alpha=0.1)

    plt.ylabel("Model Loss")
    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.legend(["Random Exploration","Optimal Exploration"])
    plt.title("Model loss against epoch")
            
    figure_name = folder_name + f"pics\\Combined median error_{metrics[m]}"
    plt.savefig(figure_name,dpi=300)

# xlabel4 = xlabel
# # torch.save(xlabel4, '..//CombinedMedianLoss//xlabel4.pt')
# # torch.save(random_median, '..//CombinedMedianLoss//LChirpDopt10Exp100Frac25Traj2RandMetrics_con_RandMedian.pt')
# # torch.save(trace_median, '..//CombinedMedianLoss//LChirpDopt10Exp100Frac25Traj2RandMetrics_con_TraceMedian.pt')

print("Plot successfully saved")