import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from binomial_tree.bt import BinomialTree

bt = BinomialTree(E=90, T=1, u=1.1, d=1/1.1, S_0=100, steps=2, typ='call')
#tree = bt.backprop(r=0, ret_tree=True)
S_tree = bt.stock_price()
print(S_tree)
