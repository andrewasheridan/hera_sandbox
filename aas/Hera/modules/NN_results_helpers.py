import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob

def plot_costs(costs, title, highlight_run = 999, ylim = [0,0.10], alt_alpha = 0.3):
    fig, ax = plt.subplots(figsize = (6,3), dpi = 144)
    
    for i in range(len(costs)):
        if costs[i][:10].mean() > 0:

            alpha = 1 if i == highlight_run else alt_alpha
            ax.plot(np.arange(len(costs[i])), costs[i], lw = 0.5, label = 'Run %d' %i, alpha = alpha)
            ax.legend(loc = 'upper right', fontsize = 5)

    ax.set_title(title)
    ax.set_ylabel('Cost')
    ax.set_xlabel('Epoch')
    ax.set_ylim(ylim[0], ylim[1])
    plt.show()    
    
def get_costs(log_direc, num_epochs):
    
    runs = glob.glob(log_direc + '*')
    training_summary_paths = [glob.glob(run + '/training/*') for run in runs]
    testing_summary_paths  = [glob.glob(run + '/testing/*')  for run in runs]
    
    training_costs = np.zeros((len(runs), num_epochs))
    testing_costs =  np.zeros((len(runs), num_epochs))

    for i in range(len(runs)):
        training_path = training_summary_paths[i]    
        training_cost = []

        for event in tf.train.summary_iterator(training_path[0]):
            for v in event.summary.value:
                if v.tag == "logging/current_cost":
                    training_cost.append(v.simple_value)

        if len(training_cost) == num_epochs:            
            training_costs[i] = training_cost

        testing_path  = testing_summary_paths[i]
        testing_cost = []
        for event in tf.train.summary_iterator(testing_path[0]):
            for v in event.summary.value:
                if v.tag == "logging/current_cost":
                    testing_cost.append(v.simple_value)
                    
        if len(testing_cost) == num_epochs: 
            testing_costs[i] = testing_cost

    return (training_costs, testing_costs)