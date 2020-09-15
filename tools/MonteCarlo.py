"""
Created: 9/15/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
import numpy as np


class MonteCarlo(object):
    def __init__(self, mu, sigma, ndata):
        self.mu = mu
        self.sigma = sigma
        self.ndata = ndata
        self.random_value()

    def random_value(self):
        data = np.random.normal(self.mu, self.sigma, self.ndata)
        return data
