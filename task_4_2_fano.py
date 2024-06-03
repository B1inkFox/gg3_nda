# import
from inference import *
from HMM_models import *
import matplotlib.pyplot as plt
import numpy as np

#common parameters
x0 = 0.2
Rh = 75
T = 100
K = 100

# define parameters ramp
beta = 2
sigma = 2

# define parameters step
m = 40
r = 5

def trial_fano(model, iterations, t):
    
    arr = np.empty([T])
    for i in range(iterations):
        latent, rate, spikes = model.simulate()
        arr = np.vstack((arr, spikes))

    mean = np.array([])
    var = np.array([])
    for i in range(t):
        sum = 0
        sqr_sum = 0
        for j in range(iterations):
            sum += arr[j][i]
            sqr_sum += arr[j][i] ** 2
        mean = np.append(mean, sum / iterations)
        var = np.append(var, sqr_sum / iterations - (sum / iterations)**2)
    fano = var / mean
    return mean, var, fano


for gamma in range(2,6):
    model = HMM_Ramp(beta, sigma, K, x0, Rh, T, isi_gamma_shape = gamma)
    mean, var, fano = trial_fano(model, 1000, T)
    spike_times = np.linspace(0, 1, num = fano.shape[0], endpoint = False)
    plt.plot(spike_times, fano, label = 'PSTH for $\gamma$ = '+str(gamma))
plt.title('Fano factor of ramp model  ' + '$\\beta$=' + str(beta) + '  $\sigma$=' + str(sigma))
plt.xlabel('time (s)')
plt.legend()
plt.show()

for gamma in range(2,6):
    model = HMM_Step(m, r, x0, Rh, T, isi_gamma_shape = gamma)
    mean, var, fano = trial_fano(model, 1000, T)
    spike_times = np.linspace(0, 1, num = fano.shape[0], endpoint = False)
    plt.plot(spike_times, fano, label = 'PSTH for $\gamma$ = '+str(gamma))
plt.title('Fano factor of step model  '+'m ='+str(m)+'  r='+str(r))
plt.xlabel('time (s)')
plt.legend()
plt.show()




#plt.plot(spike_times, mean, label = 'mean')
#plt.plot(spike_times, var, label = 'var')
#plt.plot(spike_times, fano, label = 'fano')
#plt.show()