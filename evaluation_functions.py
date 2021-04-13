import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def plot_learning_curves(history):
    """
    history is a dict containing all the information about training. Specifically the keys are:
    - train_losses;
    - train_accuracies;
    - valid_losses;
    - valid_accuracies
    """
    fig, axes = plt.subplots(1,2)
    _ = axes[0].plot(history['train_losses'], label='Train Loss')
    _ = axes[0].plot(history['valid_losses'], label = 'Validation Loss')
    axes[0].legend(loc='best')
    _ = axes[1].plot(history['train_accuracies'], label='Train Accuracy')
    _ = axes[1].plot(history['valid_accuracies'], label = 'Validation Accuracy')
    plt.legend(loc='best')

    fig.set_figheight(4)
    fig.set_figwidth(12)  
    return

def plot_losses(x):
    fig, axes = plt.subplots(len(list(x.keys())),2)
    for i,key in enumerate(list(x.keys())):
        for j in range(len(x[key]['NN'])):
            if j==0:
                axes[i,0].plot(x[key]['NN'][j]['train_losses'], c='b', alpha=0.3, label='Train')
                axes[i,0].plot(x[key]['NN'][j]['valid_losses'], c='r', alpha=0.4, label= 'Validation')
            
            axes[i,0].plot(x[key]['NN'][j]['train_losses'], c='b', alpha=0.3)
            axes[i,0].plot(x[key]['NN'][j]['valid_losses'], c='r', alpha=0.4)
            axes[i,0].legend(loc='best')
            axes[i,0].set_title('NN ({}%)'.format(key))
            axes[i,0].set_xlabel('Epochs')
            axes[i,0].set_ylabel('Loss')
            


        for j in range(len(x[key]['KENN'])):
            if j==0:
                axes[i,1].plot(x[key]['KENN'][j]['train_losses'], c='b', alpha=0.3, label='Train')
                axes[i,1].plot(x[key]['KENN'][j]['valid_losses'], c='r', alpha=0.4, label= 'Validation')
            
            axes[i,1].plot(x[key]['KENN'][j]['train_losses'], c='b', alpha=0.3)
            axes[i,1].plot(x[key]['KENN'][j]['valid_losses'], c='r', alpha=0.4)
            axes[i,1].legend(loc='best')
            axes[i,1].set_title('KENN ({}%)'.format(key))
            axes[i,1].set_xlabel('Epochs')
            axes[i,1].set_ylabel('Loss')

    fig.set_figheight(24)
    fig.set_figwidth(12)
    plt.subplots_adjust(hspace=0.3)

    plt.show()

def plot_accuracies(x):
    fig, axes = plt.subplots(len(list(x.keys())),2)
    for i,key in enumerate(list(x.keys())):
        for j in range(len(x[key]['NN'])):
            if j==0:
                axes[i,0].plot(x[key]['NN'][j]['train_accuracies'], c='b', alpha=0.3, label='Train')
                axes[i,0].plot(x[key]['NN'][j]['valid_accuracies'], c='r', alpha=0.4, label= 'Validation')
            
            axes[i,0].plot(x[key]['NN'][j]['train_accuracies'], c='b', alpha=0.3)
            axes[i,0].plot(x[key]['NN'][j]['valid_accuracies'], c='r', alpha=0.4)
            axes[i,0].legend(loc='best')
            axes[i,0].set_title('NN ({}%)'.format(key))
            axes[i,0].set_xlabel('Epochs')
            axes[i,0].set_ylabel('Accuracy')
            


        for j in range(len(x[key]['KENN'])):
            if j==0:
                axes[i,1].plot(x[key]['KENN'][j]['train_accuracies'], c='b', alpha=0.3, label='Train')
                axes[i,1].plot(x[key]['KENN'][j]['valid_accuracies'], c='r', alpha=0.4, label= 'Validation')
            
            axes[i,1].plot(x[key]['KENN'][j]['train_accuracies'], c='b', alpha=0.3)
            axes[i,1].plot(x[key]['KENN'][j]['valid_accuracies'], c='r', alpha=0.4)
            axes[i,1].legend(loc='best')
            axes[i,1].set_title('KENN ({}%)'.format(key))
            axes[i,1].set_xlabel('Epochs')
            axes[i,1].set_ylabel('Accuracy')

    fig.set_figheight(24)
    fig.set_figwidth(12)
    plt.subplots_adjust(hspace=0.3)

    plt.show()

def get_means_and_stds(history):
    means = []
    stds = []
    means_kenn = []
    stds_kenn = []

    n_runs = len(history[list(history.keys())[0]]['NN'])

    for num in history.keys():
        test_accuracies = [history[num]['NN'][i]['test_accuracy'] for i in range(n_runs)]
        mean_test_accuracies = np.mean(test_accuracies)
        std_test_accuracies = np.std(test_accuracies)

        # Append to lists
        means.append(mean_test_accuracies)
        stds.append(std_test_accuracies)

        test_accuracies_kenn = [history[num]['KENN'][i]['test_accuracy'] for i in range(n_runs)]
        mean_test_accuracies_kenn = np.mean(test_accuracies_kenn)
        std_test_accuracies_kenn = np.std(test_accuracies_kenn)

        means_kenn.append(mean_test_accuracies_kenn)
        stds_kenn.append(std_test_accuracies_kenn)
    
    return (means, stds, means_kenn, stds_kenn)

def plot_means_and_stds(history, title, barwidth=0.3):
    means, stds, means_kenn, stds_kenn = get_means_and_stds(history)
    barWidth = barwidth

    plt.figure(figsize=(9,5))
    # Set position of bar on X axis
    r1 = np.arange(len(means))
    r2 = [x + barWidth for x in r1]
    
    # Make the plot
    plt.bar(r1, means, yerr=stds , color='b', width=barWidth, edgecolor='white', label='NN')
    plt.bar(r2, means_kenn, yerr=stds_kenn, color='r', width=barWidth, edgecolor='white', label='KENN')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Percentage of Training', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(means))], history.keys())
    plt.legend(loc='best')
    plt.title(title)

    plt.show()
    return

def plot_deltas(history, barwidth=0.3, title='', other_deltas =''):
    assert(other_deltas=='' or other_deltas=='i' or other_deltas == 't')
    means, _, means_kenn, _ = get_means_and_stds(history)

    results_NN_Marra_i = np.array([0.645, 0.674, 0.707, 0.717, 0.723])
    results_SBR_i = np.array([0.650, 0.682, 0.712, 0.719, 0.726])
    results_RNM_i = np.array([0.685, 0.709, 0.726, 0.726, 0.732])

    results_NN_Marra_t = np.array([0.640, 0.667, 0.695, 0.708, 0.726])
    results_SBR_t = np.array([0.703, 0.729, 0.747, 0.764, 0.780])
    results_RNM_t = np.array([0.708, 0.735, 0.753, 0.766, 0.780])

    deltas_SBR_i = results_SBR_i - results_NN_Marra_i
    deltas_RNM_i = results_RNM_i - results_NN_Marra_i

    deltas_SBR_t = results_SBR_t - results_NN_Marra_t
    deltas_RNM_t = results_RNM_t - results_NN_Marra_t

    deltas = np.array(means_kenn) - np.array(means)
    r = np.arange(len(deltas))

    plt.figure(figsize=(9,5))
    plt.bar(r, deltas, color='b', width=barwidth, edgecolor='white', label='delta KENN')

    if other_deltas=='i':
        r2 = [x + barwidth for x in r]
        r3 = [x + barwidth for x in r2]
        plt.bar(r2, deltas_SBR_i, color='r', width=barwidth, edgecolor='white', label='delta SBR')
        plt.bar(r3, deltas_RNM_i, color='g', width=barwidth, edgecolor='white', label='delta RNM')

    elif other_deltas=='t':
        r2 = [x + barwidth for x in r]
        r3 = [x + barwidth for x in r2]
        plt.bar(r2, deltas_SBR_t, color='r', width=barwidth, edgecolor='white', label='delta SBR')
        plt.bar(r3, deltas_RNM_t, color='g', width=barwidth, edgecolor='white', label='delta RNM')

    plt.xlabel('Percentage of Training', fontweight='bold')
    plt.xticks([r + barwidth for r in range(len(means))], list(history.keys()))
    plt.legend(loc='best')
    plt.title(title)
    return

def print_stats(history):
    means, stds, means_kenn, stds_kenn = get_means_and_stds(history)

    for i,key in enumerate(history.keys()):
        print("== {}% ==".format(key))
        print("Mean Test Accuracy:\tNN = {:8.6f}; KENN = {:8.6f}".format(means[i], means_kenn[i]))
        print("Test Accuracy std:\tNN = {:8.6f}; KENN = {:8.6f}".format(stds[i], stds_kenn[i]))
        print("\t\t\tDelta = {:8.6f}".format(means_kenn[i]-means[i]))
        print()

def print_and_plot_results(history, plot_title, other_deltas=''):
    """
    Parameters:
    - other_deltas: a string taking values in ['', 'i', 't']. 
        - '': Only the deltas from kenn are plotted:
        - 'i': The deltas from the other inductive experiments are printed along our deltas
        - 't': The deltas from the other transductive experiments are printed along our deltas
    """
    means, stds, means_kenn, stds_kenn = get_means_and_stds(history)
    print_stats(history)
    plot_means_and_stds(history, plot_title, 0.4)
    plot_deltas(history, title=plot_title, other_deltas=other_deltas)
    return