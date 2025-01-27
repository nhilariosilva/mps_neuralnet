import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from time import time

from scipy.special import comb, loggamma, lambertw
from scipy.stats import multinomial, expon

import lifelines

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

import os, shutil
import json
import subprocess

import mps
import pwexp
from net_model import *
from mps_models import *


# Loading dataset

print("Loading Fashion-MNIST dataset...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

i_valid_train = pd.Series(train_labels).isin([0,1,2,3,4]).to_numpy()
i_valid_test = pd.Series(test_labels).isin([0,1,2,3,4]).to_numpy()

# Filters to take only the images with labels in [0, 1, 2, 3, 4]
train_labels = train_labels[i_valid_train]
test_labels = test_labels[i_valid_test]

# Indices for each set of filtered data
i_train = np.arange(train_images.shape[0])
i_test = np.arange(test_images.shape[0])

print("Dimension of the train set: {}".format(train_images.shape))
print("Dimension of the test set: {}".format(test_images.shape))

print("Done...")

# Definição das probabilidades (REMOVER QUANDO FINALIZADOS OS CODIGOS)
probs1 = np.array([0.9, 0.45, 0.22, 0.14, 0.08])
cure_probs1 = pd.DataFrame({"Classe": ["0", "1", "2", "3", "4"], "p_0": [0.9, 0.45, 0.22, 0.14, 0.08]})
cure_probs_dict1 = {0: 0.9, 1:0.45, 2:0.22, 3:0.14, 4: 0.08}
cure_probs_dict1 = np.vectorize(cure_probs_dict1.get)
cure_probs1

def get_theta(log_a, log_phi, C, C_inv, sup, p_0, theta_min = None, theta_max = None):
    '''
        Given the specifications for the latent causes distribution and a vector with cure probabilities,
        inverts the cure probability function and returns the theta parameters for each individual
    '''
    theta = C_inv( np.exp(log_a(0.0) - np.log(p_0)) )
    
    # Se theta é limitado inferiormente por um valor theta_min > 0, valores de theta obtidos abaixo do limite são levados para o limite inferior do parâmetro
    if(theta_min is not None):
        theta[theta <= theta_min] = theta_min + 1.0e-5
    # Se theta é limitado superiormente por um valor theta_max > 0, valores de theta obtidos acima do limite são levados para o limite superior do parâmetro
    if(theta_min is not None):
        theta[theta >= theta_max] = theta_max - 1.0e-5
        
    return theta


def generate_data(log_a, log_phi, theta, sup, low_c, high_c):
    '''
        Dada a especificação do modelo e um vetor com os parâmetros individuais, gera os tempos de vida e censuras de cada indivíduo.
        low_c e high_c definem o intervalo para a geração dos tempos de censura, seguindo uma distribuição U[low_c, high_c]
    '''
    n = len(theta)
    m = mps.rvs(log_a, log_phi, theta, sup, size = 10)
    
    cured = np.zeros(n)
    delta = cured.copy()
    t = cured.copy()
    
    # Censorship times
    c = np.random.uniform(low = low_c, high = high_c, size = n)
    
    for i in range(n):
        if(m[i] == 0):
            t[i] = c[i]
            cured[i] = 1
        else:
            # Risco base segue uma distribuição Exp(1)
            z = expon.rvs(loc = 0.0, scale = 1.0, size = int(m[i]))
            t[i] = np.min(z)
    
    # Atualiza as posições não censuradas para delta = 1
    delta[t < c] = 1
    # Os tempos censurados passam a assumir o valor do tempo de censura
    t[t >= c] = c[t >= c]
    
    # Retorna os tempos, deltas e o vetor de causas latentes (que na prática é desconhecido)
    return m, t, delta, cured



def sample_bootstrap_scenario1(n_train, n_val, n_test, cure_probs_dict, directory):
    n = n_train + n_val + n_test

    # ---------------------------- Sample the indices from the original dataset ----------------------------
    
    # Indices for train and validation
    i_train_val = np.random.choice(i_train, size = n_train + n_val, replace = True)
    i_test = np.random.choice(i_test, size = n_test, replace = True)
    
    # The labels for the train set are the first n_train sampled indices in i_train_val
    label_train = train_labels[i_train_val[:n_train]]
    # The labels for the validation set are the last n_train sampled indices in i_train_val
    label_val = train_labels[i_train_val[n_train:]]
    # Takes the labels for the test set
    label_test = test_labels[i_test]
    
    p_train = cure_probs_dict(label_train)
    p_val = cure_probs_dict(label_val)
    p_test = cure_probs_dict(label_test)

    # ---------------------------- Poisson ----------------------------
    # Poisson - Training data
    theta_train_poisson = get_theta(log_a_poisson, log_phi_poisson, C_poisson, C_inv_poisson, sup_poisson, p_train, theta_min = theta_min_poisson, theta_max = theta_max_poisson)
    m_train_poisson, t_train_poisson, delta_train_poisson, cured_train_poisson = \
        generate_data(log_a_poisson, log_phi_poisson, theta_train_poisson, sup_poisson, low_c, high_c)
    # Poisson - Validation data
    theta_val_poisson = get_theta(log_a_poisson, log_phi_poisson, C_poisson, C_inv_poisson, sup_poisson, p_val, theta_min = theta_min_poisson, theta_max = theta_max_poisson)
    m_val_poisson, t_val_poisson, delta_val_poisson, cured_val_poisson = \
        generate_data(log_a_poisson, log_phi_poisson, theta_val_poisson, sup_poisson, low_c, high_c)
    # Poisson - Test data
    theta_test_poisson = get_theta(log_a_poisson, log_phi_poisson, C_poisson, C_inv_poisson, sup_poisson, p_test, theta_min = theta_min_poisson, theta_max = theta_max_poisson)
    m_test_poisson, t_test_poisson, delta_test_poisson, cured_test_poisson = \
        generate_data(log_a_poisson, log_phi_poisson, theta_test_poisson, sup_poisson, low_c, high_c)
    
    # ---------------------------- Logarithmic ----------------------------
    # Logarithmic - Training data
    theta_train_log = get_theta(log_a_log, log_phi_log, C_log, C_inv_log, sup_log, p_train, theta_min = theta_min_log, theta_max = theta_max_log)
    m_train_log, t_train_log, delta_train_log, cured_train_log = \
        generate_data(log_a_log, log_phi_log, theta_train_log, sup_log, low_c, high_c)
    # Logarithmic - Validation data
    theta_val_log = get_theta(log_a_log, log_phi_log, C_log, C_inv_log, sup_log, p_val, theta_min = theta_min_log, theta_max = theta_max_log)
    m_val_log, t_val_log, delta_val_log, cured_val_log = \
        generate_data(log_a_log, log_phi_log, theta_val_log, sup_log, low_c, high_c)
    # Logarithmic - Test data
    theta_test_log = get_theta(log_a_log, log_phi_log, C_log, C_inv_log, sup_log, p_test, theta_min = theta_min_log, theta_max = theta_max_log)
    m_test_log, t_test_log, delta_test_log, cured_test_log = \
        generate_data(log_a_log, log_phi_log, theta_test_log, sup_log, low_c, high_c)

    # ---------------------------- Geometric ----------------------------
    # Geometric - Training data
    theta_train_geo = get_theta(log_a_nb(1), log_phi_nb(1), C_nb(1), C_inv_nb(1), sup_nb, p_train, theta_min = theta_min_nb, theta_max = theta_max_nb)
    m_train_geo, t_train_geo, delta_train_geo, cured_train_geo = \
        generate_data(log_a_nb(1), log_phi_nb(1), theta_train_geo, sup_nb, low_c, high_c)
    # Geometric - Validation data
    theta_val_geo = get_theta(log_a_nb(1), log_phi_nb(1), C_nb(1), C_inv_nb(1), sup_nb, p_val, theta_min = theta_min_nb, theta_max = theta_max_nb)
    m_val_geo, t_val_geo, delta_val_geo, cured_val_geo = \
        generate_data(log_a_nb(1), log_phi_nb(1), theta_val_geo, sup_nb, low_c, high_c)
    # Geometric - Test data
    theta_test_geo = get_theta(log_a_nb(1), log_phi_nb(1), C_nb(1), C_inv_nb(1), sup_nb, p_test, theta_min = theta_min_nb, theta_max = theta_max_nb)
    m_test_geo, t_test_geo, delta_test_geo, cured_test_geo = \
        generate_data(log_a_nb(1), log_phi_nb(1), theta_test_geo, sup_nb, low_c, high_c)
    
    # ---------------------------- NB(q = 2) ----------------------------
    # NB(2) - Training data
    theta_train_2nb = get_theta(log_a_nb(2), log_phi_nb(2), C_nb(2), C_inv_nb(2), sup_nb, p_train, theta_min = theta_min_nb, theta_max = theta_max_nb)
    m_train_2nb, t_train_2nb, delta_train_2nb, cured_train_2nb = \
        generate_data(log_a_nb(1), log_phi_nb(1), theta_train_2nb, sup_nb, low_c, high_c)
    # NB(2) - Validation data
    theta_val_2nb = get_theta(log_a_nb(2), log_phi_nb(2), C_nb(2), C_inv_nb(2), sup_nb, p_val, theta_min = theta_min_nb, theta_max = theta_max_nb)
    m_val_2nb, t_val_2nb, delta_val_2nb, cured_val_2nb = \
        generate_data(log_a_nb(2), log_phi_nb(2), theta_val_2nb, sup_nb, low_c, high_c)
    # NB(2) - Test data
    theta_test_2nb = get_theta(log_a_nb(2), log_phi_nb(2), C_nb(2), C_inv_nb(2), sup_nb, p_test, theta_min = theta_min_nb, theta_max = theta_max_nb)
    m_test_2nb, t_test_2nb, delta_test_2nb, cured_test_2nb = \
        generate_data(log_a_nb(2), log_phi_nb(2), theta_test_2nb, sup_nb, low_c, high_c)
    
    # ---------------------------- Bernoulli ----------------------------
    # Bernoulli - Training data
    theta_train_bern = get_theta(log_a_bin(1), log_phi_bin(1), C_bin(1), C_inv_bin(1), sup_bin(1), p_train, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_train_bern, t_train_bern, delta_train_bern, cured_train_bern = \
        generate_data(log_a_bin(1), log_phi_bin(1), theta_train_bern, sup_bin(1), low_c, high_c)
    # Bernoulli - Validation data
    theta_val_bern = get_theta(log_a_bin(1), log_phi_bin(1), C_bin(1), C_inv_bin(1), sup_bin(1), p_val, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_val_bern, t_val_bern, delta_val_bern, cured_val_bern = \
        generate_data(log_a_bin(1), log_phi_bin(1), theta_val_bern, sup_bin(1), low_c, high_c)
    # Bernoulli - Test data
    theta_test_bern = get_theta(log_a_bin(1), log_phi_bin(1), C_bin(1), C_inv_bin(1), sup_bin(1), p_test, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_test_bern, t_test_bern, delta_test_bern, cured_test_bern = \
        generate_data(log_a_bin(1), log_phi_bin(1), theta_test_bern, sup_bin(1), low_c, high_c)

    # ---------------------------- Binomial (q = 5) ----------------------------
    # Binomial(5) - Training data
    theta_train_5bin = get_theta(log_a_bin(5), log_phi_bin(5), C_bin(5), C_inv_bin(5), sup_bin(5), p_train, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_train_5bin, t_train_5bin, delta_train_5bin, cured_train_5bin = \
        generate_data(log_a_bin(5), log_phi_bin(5), theta_train_5bin, sup_bin(5), low_c, high_c)
    # Binomial(5) - Validation data
    theta_val_5bin = get_theta(log_a_bin(5), log_phi_bin(5), C_bin(5), C_inv_bin(5), sup_bin(5), p_val, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_val_5bin, t_val_5bin, delta_val_5bin, cured_val_5bin = \
        generate_data(log_a_bin(5), log_phi_bin(5), theta_val_5bin, sup_bin(5), low_c, high_c)
    # Binomial(5) - Test data
    theta_test_5bin = get_theta(log_a_bin(5), log_phi_bin(5), C_bin(5), C_inv_bin(5), sup_bin(5), p_test, theta_min = theta_min_bin, theta_max = theta_max_bin)
    m_test_5bin, t_test_5bin, delta_test_5bin, cured_test_5bin = \
        generate_data(log_a_bin(5), log_phi_bin(5), theta_test_5bin, sup_bin(5), low_c, high_c)
    
    
    
    
