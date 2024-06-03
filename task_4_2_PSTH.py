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
m = 60
r = 5

def trial_average(model, iterations, t, N):
    
    bin = np.zeros(t)
    for i in range(iterations):
        latent, rate, spikes = model.simulate()
        bin += spikes
    bin  = bin / iterations
    
    bin = np.convolve(bin, np.ones(N)/N, mode='valid')
    return bin

for gamma in range(2,6):
    model = HMM_Ramp(beta, sigma, K, x0, Rh, T, isi_gamma_shape = gamma)
    bin = trial_average(model, 1000, T, 5)
    spike_times = np.linspace(0, 1, num = bin.shape[0], endpoint = False)
    plt.plot(spike_times, bin, label = 'PSTH for $\gamma$ = '+str(gamma))
plt.title('PSTH of ramp model  ' + '$\\beta$=' + str(beta) + '  $\sigma$=' + str(sigma))
plt.xlabel('time (s)')
plt.legend()
plt.show()

for gamma in range(2,6):
    model = HMM_Step(m, r, x0, Rh, T, isi_gamma_shape = gamma)
    bin = trial_average(model, 1000, T, 5)
    spike_times = np.linspace(0, 1, num = bin.shape[0], endpoint = False)
    plt.plot(spike_times, bin, label = 'PSTH for $\gamma$ = '+str(gamma))
plt.title('PSTH of step model  '+'m ='+str(m)+'  r='+str(r))
plt.xlabel('time (s)')
plt.legend()
plt.show()