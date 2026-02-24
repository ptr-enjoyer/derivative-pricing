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

    def __init__(self, E, T, mu, sigma, S_0, steps, typ, contract='European', dividends=False):
        super().__init__(E, T, typ, contract=contract)
        self.steps = steps
        self.u = np.exp(sigma*np.sqrt(self.T / self.steps))
        self.d = 1 / self.u
        self.S_0 = S_0

        self.dividends = dividends  or {}



    def calc_delta(self, V_p, V_m, S):
        delta = (V_p - V_m) / (self.u*S - self.d*S)
        return delta 
    
    def option_val(self, r, V_p, V_m):
        t = self.T / self.steps
        p = (np.exp((r*t)) - self.d) / (self.u-self.d)
        V = np.exp((-r*t)) * (p * V_p + (1-p)* V_m)
        return V
    
    def stock_price(self):

    
        S = [[float(self.S_0)]]

        for i in range(1, self.steps + 1):
            prev = S[i - 1]
            level = [0.0] * (i + 1)

            level[0] = prev[0] * self.d
            for j in range(1, i):
                level[j] = prev[j - 1] * self.u
            level[i] = prev[i - 1] * self.u

            # Apply cash dividend drop AT THIS STEP (ex-div)
            if self.dividends and i in self.dividends:
                D = float(self.dividends[i])
                level = [round(max(s - D, 0.0)) for s in level]

            S.append(level)

        return S
        """
            S = []
            for i in range(self.steps+1):
                depth = []
                for j in range(i+1):
                    depth.append(round(self.S_0 * (self.u**j ) * (self.d**(i-j)), 4))
                S.append(depth)
            return S
        """
        


    def backprop(self, r, ret_tree=False):
        t = self.T / self.steps
        disc = np.exp(-r * t)
        p = (np.exp(r * t) - self.d ) / (self.u - self.d) # Risk-accounted probability of an "up" move
        V_prices = []
        S_prices = self.stock_price()
        depth = len(S_prices)

        for i in range(1, depth+1):
            V_prices.append([0]*i)
        
        exp_V = V_prices[-1]
        exp_S = S_prices[-1]

        for i in range(len(exp_V)):
            exp_V[i] = self.option_exp(exp_S[i])

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
        
        if ret_tree:
            return V_prices
        
        return round(V_prices[0][0], 2)
    
    def plot_binomial_tree(self, stock_tree, option_tree, title='Binomial Tree', x_gap=1, y_gap=1, offset=0.07, annotate=True):
        n = len(option_tree)
        if n == 0:
            raise ValueError('tree is empty')
        
        xs = []
        ys = []
        labels = []

        coords = {}
        for i in range(n):
            m = len(option_tree[i])
            for j in range(m):
                x = i * x_gap
                y = (j - (m-1) / 2) * y_gap
                coords[(i, j)] = (x, y)
                xs.append(x)
                ys.append(y)
                labels.append(round(option_tree[i][j], 4)) 
        
        fig, ax = plt.subplots()
        ax.set_title(title)

        for i in range(n-1):
            for j in range(len(option_tree[i])):
                x0, y0 = coords[(i, j)]
                x1, y1 = coords[(i+1, j)]
                x2, y2 = coords[(i+1, j+1)]
                ax.plot([x0, x1], [y0, y1], linewidth=1)
                ax.plot([x0, x2], [y0, y2], linewidth=1)
        if annotate:
            for (i, j), (x, y) in coords.items():
                    S = stock_tree[i][j]
                    V = option_tree[i][j]
                    ax.text(x, y + offset, f"{S:.2f}", ha="center", va="bottom", fontsize=8, zorder=4)
                    ax.text(x, y - offset, f"{V:.2f}", ha="center", va="top", fontsize=8, zorder=4)



        ax.axis("off")
        plt.show()

        

