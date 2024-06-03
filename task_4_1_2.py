# import
from inference import *
from HMM_models import *
from HMM_inference import *
import numpy as np
from scipy.stats import truncnorm


# Truncated Gaussian prior
beta_center = beta_space[0]
sigma_center = (sigma_space[0] + sigma_space[-1]) / 2
m_center = m_space[0]
r_center = (r_space[0]+r_space[-1]) / 2

fraction = 0.2
beta_std = fraction * (beta_space[-1] - beta_space[0])
sigma_std = fraction * (sigma_space[-1] - sigma_space[0])
m_std = fraction * (m_space[-1] - m_space[0])
r_std = fraction * (r_space[-1] - r_space[0])

def truncated_gaussian_prior(mean, std_dev, lower, upper, size=1):
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    return truncnorm(a, b, loc=mean, scale=std_dev).rvs(size)

beta_prior = truncated_gaussian_prior(beta_center, beta_std, beta_space[0], beta_space[-1], size = len(beta_space))
sigma_prior = truncated_gaussian_prior(sigma_center, sigma_std, sigma_space[0], sigma_space[-1], size = len(sigma_space))
m_prior = truncated_gaussian_prior(m_center, m_std, m_space[0], m_space[-1], size = len(m_space))
r_prior = truncated_gaussian_prior(r_center, r_std, r_space[0], r_space[-1], size = len(r_space))

for p in [beta_prior, sigma_prior, m_prior, r_prior]:
    p /= np.sum(p)

ramp_priori_gauss = np.outer(beta_prior, sigma_prior)
step_priori_gauss = np.outer(m_prior, r_prior)


ramp_priori_uniform = np.ones([len(beta_space), len(sigma_space)]) / (len(beta_space)*len(sigma_space))
step_priori_uniform = np.ones([len(m_space), len(r_space)]) / (len(m_space)*len(r_space))



gamma = 2 #gamma = 1 to 5
iter = 100
N = 25

result_step = np.empty([])
error_step = 0
for i in range(iter):
    m,r = sample_from_priori(step_priori_gauss, model = 'step')
    shmm = HMM_Step(m, r, x0, Rh, T, isi_gamma_shape=gamma)
    shmm_datas = generate_N_trials(N, shmm)
    bayes = compute_bayes_factor(shmm_datas, ramp_priori_uniform, step_priori_uniform)
    result_step = np.append(result_step, bayes)
    if bayes < 0:
        error_step += 1
#print('Step decision results', result_step)
print('Step decision accuracy', error_step / iter)

result_ramp = np.empty([])
error_ramp = 0
for i in range(iter):
    beta, sigma = sample_from_priori(ramp_priori_gauss, model = 'ramp')
    rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T, isi_gamma_shape=gamma)
    rhmm_datas = generate_N_trials(N, rhmm)
    bayes = compute_bayes_factor(rhmm_datas, ramp_priori_uniform, step_priori_uniform)
    result_ramp = np.append(result_ramp, bayes)
    if bayes > 0:
        error_ramp += 1
#print('Ramp decision results', result_ramp)
print('Ramp decision accuracy', error_ramp / iter)
