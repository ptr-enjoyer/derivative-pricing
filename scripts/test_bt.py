import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from binomial_tree.bt import BinomialTree

bt = BinomialTree(E=110, T=1, mu=1.1, sigma=0.3, S_0=100, steps=5, typ='call')

S_tree = bt.stock_price()
#print(S_tree)

tree = bt.backprop(r=0)
print(tree)
bt.plot_binomial_tree(S_tree, bt.backprop(r=0, ret_tree=True))


