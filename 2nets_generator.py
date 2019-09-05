import numpy as np

##########################################
# Generate topologies for neural networks#
##########################################

def generador(red_1, n_redes = 1):
    '''Generates n_redes from original net'''
    n_1 = np.array(red_1)
    n_12 =np.copy(n_1)
    for x in range(0, n_redes-1):
        n_12 = n_12*np.array([2])
        n_1 = np.append(n_1, n_12)
    l = len(red_1)    
    n_1 = np.reshape(n_1, (n_redes, l))
    return n_1

#PIRAMID:
a1 = generador([50, 25, 12, 4], 10)
print(a1)

#Flipped PIRAMID:
a2 = generador([4, 12, 25, 50], 10)
print(a2)
    