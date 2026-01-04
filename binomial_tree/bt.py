import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BinomialTree():

    def __init__(self, E, T, u, d, steps, contract='European'):
        self.E = E
        self.T = T
        self.u = u
        self.d = d
        self.steps = steps
        self.contract = contract
    
    def calc_delta(self, V_p, V_m, S):
        delta = (V_p - V_m) / (self.u*S - self.d*S)
        return delta
    
    def option_val(self, r, V_p, V_m):
        t = self.T / self.steps
        p = (np.exp ** (r*t) - self.d) / (self.u-self.d)
        V = np.exp **(-r*t) * (p * V_p + (1-p)* V_m)
        return V