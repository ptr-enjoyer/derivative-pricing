import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BinomialTree():

    def __init__(self, E, T, u, d, contract='European'):
        self.E = E
        self.T = T
        self.u = u
        self.d = d
        self.contract = contract
    
    def calc_delta(self, V_p, V_m, S):
        delta = (V_p - V_m) / (self.u*S - self.d*S)
        return delta 