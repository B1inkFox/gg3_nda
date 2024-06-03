# import
from inference import *
from HMM_models import *
from HMM_inference import *
import numpy as np


ramp_priori = np.ones([len(beta_space), len(sigma_space)]) / (len(beta_space)*len(sigma_space))
step_priori = np.ones([len(m_space), len(r_space)]) / (len(m_space)*len(r_space))



iter = 100
N = 5

result_step = np.empty([])
error_step = 0
for i in range(iter):
    m,r = sample_from_priori(step_priori, model = 'step')
    shmm = HMM_Step(m, r, x0, Rh, T)
    shmm_datas = generate_N_trials(N, shmm)
    bayes = compute_bayes_factor(shmm_datas, ramp_priori, step_priori)
    result_step = np.append(result_step, bayes)
    if bayes < 0:
        error_step += 1
#print('Step decision results', result_step)
print('Step decision accuracy', error_step / iter)

result_ramp = np.empty([])
error_ramp = 0
for i in range(iter):
    beta, sigma = sample_from_priori(ramp_priori, model = 'ramp')
    rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
    rhmm_datas = generate_N_trials(N, rhmm)
    bayes = compute_bayes_factor(rhmm_datas, ramp_priori, step_priori)
    result_ramp = np.append(result_ramp, bayes)
    if bayes > 0:
        error_ramp += 1
#print('Ramp decision results', result_ramp)
print('Ramp decision accuracy', error_ramp / iter)