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

    def backprop(self, r, ret_tree=False):
        disc = np.exp(-r*self.T/self.steps)
        p = (disc - 1/self.u ) / (self.u-1/self.u) # Risk-accounted probability of an "up" move
        V_prices = []
        S_prices = self.stock_price()
        depth = len(S_prices)

        for i in range(1, depth+1):
            V_prices.append([0]*i)
        
        exp_V = V_prices[-1]
        exp_S = S_prices[-1]

        for i in range(len(exp_V)):
            exp_V[i] = round(self.option_exp(exp_S[i]),4)

        for i in range(depth-2, -1, -1):
            for j in range(len(S_prices[i])):
                V_curr = disc*(p*V_prices[i+1][j+1] + (1-p)*V_prices[i+1][j])
                if self.contract == 'European':
                    V_prices[i][j] = V_curr
                elif self.contract == 'American':
                    V_maybe = self.option_exp(S_prices[i][j], self.E, self.typ)
                    if V_maybe > V_curr:
                        V_prices[i][j] = V_maybe
                    else:
                        V_prices[i][j] = V_curr
        
        if ret_tree == 'True':
            return V_prices
        
        return V_prices[0][0]

