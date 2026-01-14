import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Option(): 
    def __init__(self, E, T, typ, contract='European'):
        if typ == 'call' or typ == 'put':
            self.typ = typ
        else:
            raise ValueError("typ must be either 'call' or 'put'")
        if contract == 'European' or contract == 'American':
            self.contract = contract 
        else:
            raise ValueError("contract must be either 'European' or 'American'")
        self.E = E
        self.T = T
    def option_exp(self, S_fin):
        if self.typ == 'call':
            return max(S_fin - self.E, 0)
        else:
            return max(self.E - S_fin, 0)

        



class BinomialTree(Option):

    def __init__(self, E, T, u, d, S_0, steps, typ, contract='European'):
        super().__init__(E, T, typ, contract="European")
        self.u = u
        self.d = d
        self.S_0 = S_0
        self.steps = steps

    def calc_delta(self, V_p, V_m, S):
        delta = (V_p - V_m) / (self.u*S - self.d*S)
        return delta
    
    def option_val(self, r, V_p, V_m):
        t = self.T / self.steps
        p = (np.exp ** (r*t) - self.d) / (self.u-self.d)
        V = np.exp **(-r*t) * (p * V_p + (1-p)* V_m)
        return V
    
    def stock_price(self):
        S = []
        for i in range(self.steps):
            depth = []
            for j in range(i):
                depth.append(self.S_0 * (self.u**j ) * (self.d**(i-j)))
            S.append(depth)
        return S
    

        
        