
import warnings

import numpy as np

def pmf(x, log_a, log_phi, theta, sup, force_broadcasting = False):
    # Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list] ):
        theta = np.array([theta])
    # Se x é uma lista, o converte para np.array
    if(type(x) == list):
        x = np.array(x)
    # Se theta é uma lista, o converte para np.array
    if(type(theta) == list):
        theta = np.array(theta)
    
    # Garante um formato de colunas para theta para realizar o broadcasting, caso necessário
    theta = np.reshape(theta, (len(theta), 1))
    
    # Evita problemas nas funções log_a e log_phi
    sup = sup.astype("float64") 
    
    # Obtém os valores do núcleo para o suporte da distribuição
    Psup = np.exp( log_a(sup) + sup * log_phi(theta) )
    # Obtém os valores do núcleo para o vetor x desejado
    Px = np.exp( log_a(x) + x * log_phi(theta) )
    # Normaliza o vetor de probabilidades com base na soma das probabilidades do suporte
    Px = Px / np.sum(Psup, axis = 1).reshape((len(theta),1))
    
    # Se a matriz é quadrada, subentende-se que se deseja vetorizar o cálculo, considerando um valor de theta para cada valor de x
    if(len(x) == len(theta) and not force_broadcasting):
        return np.diag(Px)
    # Caso theta seja um único número, evita o retorno de uma matriz desnecessariamente
    if(len(theta) == 1):
        Px = Px[0,:]
    
    return Px

def cdf(x, log_a, log_phi, theta, sup, lower_tail = True, force_broadcasting = False):
    # Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list] ):
        theta = np.array([theta])
    # Se x é uma lista, o converte para np.array
    if(type(x) == list):
        x = np.array(x)
    # Se theta é uma lista, o converte para np.array
    if(type(theta) == list):
        theta = np.array(theta)# Se x é um número e não um vetor
    if( type(x) not in [type(np.array([])), list] ):
        x = np.array([x])
    # Se theta é um número e não um vetor
    if( type(theta) not in [type(np.array([])), list] ):
        theta = np.array([theta])
    
    # Evita problemas nas funções log_a e log_phi
    sup = sup.astype("float64")
    # Probabilidades de cada elemento do suporte
    fsup = pmf(sup, log_a, log_phi, theta, sup)
    
    # Se len(theta) = 1, aumenta a dimensão do objeto para uma matriz de modo a facilitar as operações gerais
    if(len(theta) == 1):
        fsup = np.array([fsup.tolist()])
    
    # Probabilidades acumuladas de cada elemento do suporte
    fsup_cum = np.cumsum( fsup, axis = 1 )
    
    # Índices de cada c referente aos valores de theta
    i = np.repeat( np.arange(len(theta)), len(x) )
    # Índices de cada x referentes aos elementos do suporte
    j = np.tile( np.searchsorted(sup, x), len(theta) )
    
    fsup_cum_cdf = np.reshape( fsup_cum[i,j], (len(theta), len(x)) )
    
    if(not lower_tail):
        fsup_cum_cdf = 1-fsup_cum_cdf
    
    if(len(x) == len(theta) and not force_broadcasting):
        return np.diag(fsup_cum_cdf)
    
    # Se len(theta) = 1, retorna a matriz calculada para um vetor
    if(len(theta) == 1):
        fsup_cum_cdf = fsup_cum_cdf[0,:]
    
    return fsup_cum_cdf

def rvs_single_theta(log_a, log_phi, theta, sup):
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    probs = pmf(sup, log_a, log_phi, theta, sup)
    return np.random.choice(sup, size = 1, replace = True, p = probs)
    
def rvs(log_a, log_phi, theta, sup, size = 1):
    if( (type(theta) == list or type(theta) == type(np.array([]))) ):
        return np.array([rvs_single_theta(log_a, log_phi, the, sup) for the in theta]).flatten()
    
    probs = pmf(sup, log_a, log_phi, theta, sup)
    return np.random.choice(sup, size = size, replace = True, p = probs)

def ppf(q, log_a, log_phi, theta, sup):
    sup = sup.astype("float64") # Evita problemas nas funções a e phi
    Fs = cdf(sup, log_a, log_phi, theta, sup)
    i = np.searchsorted(Fs, q)
    return sup[i]

