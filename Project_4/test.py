import matplotlib.pyplot as plt
import numpy as np

def plot_loss_functions(zero_BP_layer, one_BP_layer, two_BP_layer,
                        zero_GA_layer, one_GA_layer, two_GA_layer,
                        zero_DE_layer, one_DE_layer, two_DE_layer,
                        zero_PS_layer, one_PS_layer, two_PS_layer,) -> None:
    '''Creates a figure with four subplots showing our results'''
    # Sets up plot for displaying 
    fig, ax = plt.subplots(4, 3, figsize=(12,10))
    fig.tight_layout(pad=3.0)
    cmap = plt.get_cmap('tab10')
    plt.subplots_adjust(left=0.16)

    #-------------------------------------------------------
    ax[0][0].set_ylabel('Backpropogation')
    ax[0][0].bar(x= range(10),
          height=[x for x in zero_BP_layer],
          color=cmap.colors, 
          width=0.5)
    
    ax[0][1].bar(x= range(10),
          height=[x for x in one_BP_layer],
          color=cmap.colors, 
          width=0.5)
    
    ax[0][2].bar(x= range(10),
          height=[x for x in two_BP_layer],
          color=cmap.colors, 
          width=0.5)
    
    ax[0][0].set_ylabel('Backpropogation', size=12)
    ax[0][0].set_title('No Hidden Layers')
    ax[0][1].set_title('One Hidden Layer')
    ax[0][2].set_title('Two Hidden Layers')

    #-------------------------------------------------------
    
    ax[1][0].bar(x= range(10),
        height=[x for x in zero_GA_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[1][1].bar(x= range(10),
        height=[x for x in one_GA_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[1][2].bar(x= range(10),
        height=[x for x in two_GA_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[1][0].set_ylabel('Genetic Algorithm', size=12)
    ax[1][0].set_title('No Hidden Layers')
    ax[1][1].set_title('One Hidden Layer')
    ax[1][2].set_title('Two Hidden Layers')

    #-------------------------------------------------------
    
    ax[2][0].bar(x= range(10),
        height=[x for x in zero_DE_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[2][1].bar(x= range(10),
        height=[x for x in one_DE_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[2][2].bar(x= range(10),
        height=[x for x in two_DE_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[2][0].set_ylabel('Differential Evolution', size=12)
    ax[2][0].set_title('No Hidden Layers')
    ax[2][1].set_title('One Hidden Layer')
    ax[2][2].set_title('Two Hidden Layers')

    #-------------------------------------------------------
    
    ax[3][0].bar(x= range(10),
        height=[x for x in zero_PS_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[3][1].bar(x= range(10),
        height=[x for x in one_PS_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[3][2].bar(x= range(10),
        height=[x for x in two_PS_layer],
        color=cmap.colors, 
        width=0.5)
    
    ax[3][0].set_ylabel('Particle Swarm', size=12)
    ax[3][0].set_title('No Hidden Layers')
    ax[3][1].set_title('One Hidden Layer')
    ax[3][2].set_title('Two Hidden Layers')
    

    labels = {}
    for i, col in enumerate(cmap.colors):
        labels['Fold ' + str(i+1)] = col
    handles = [plt.Rectangle((0,0),1,1, color=labels[label]) for label in labels.keys()]

    fig.legend(handles, labels.keys(), loc='center left')

    plt.show()

one = [np.random.rand() for _ in range(10)]

plot_loss_functions(one, one, one, one, one, one, one, one, one, one, one, one)