import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from binomial_tree.bt import BinomialTree
from binomial_tree.bt import Option
import unittest
import random

class Test_Option(unittest.TestCase):

    def test_contract_invalid(self):
        with self.assertRaises(ValueError) as cm:
            Option(10, 10, 'call', contract='Foo')
        self.assertIn('contract must be either', str(cm.exception))

    def test_contract_valid(self):
        o1 = Option(10, 10, 'call', contract='European')
        self.assertEqual(o1.contract, 'European')

        o2 = Option(10, 10, 'call', contract='American')
        self.assertEqual(o2.contract, 'American')
    

    def test_typ_invalid(self):
        with self.assertRaises(ValueError) as cm:
            Option(random.random(), random.random(), typ='boo', contract='European')
        self.assertIn('typ must be either', str(cm.exception))

    def test_typ_valid(self):
        o1 = Option(random.random(), random.random(), typ='call', contract='European')
        self.assertEqual(o1.typ, 'call')

        o2 = Option(10, 10, typ='put', contract='European')
        self.assertEqual(o2.typ, 'put')

    
    def test_option_exp_call(self):
        o = Option(E=100, T=random.random(), typ='call')
        self.assertEqual(o.option_exp(120), 20)
        self.assertEqual(o.option_exp(100), 0)
        self.assertEqual(o.option_exp(90), 0)

    def test_option_exp_put(self):
        o = Option(E=100, T=random.random(), typ='put')
        self.assertEqual(o.option_exp(120), 0)
        self.assertEqual(o.option_exp(100), 0)
        self.assertEqual(o.option_exp(90), 10)


class TestbBinomialTree(unittest.TestCase):

    # The following six tests are check the inherited instantiations of variables and methods

    def test_bt_contract_invalid(self):
        with self.assertRaises(ValueError) as cm:
            BinomialTree(random.random(), random.random(), random.random(), random.random(), 
                         random.random(), int(random.random()), typ='call', contract='Foo')
        self.assertIn('contract must be either', str(cm.exception))

    def test_bt_contract_valid(self):
        bt1 = BinomialTree(random.random(), random.random(), random.random(), random.random(), 
                         random.random(), int(random.random()), typ='call', contract='European')
        self.assertEqual(bt1.contract, 'European')

        bt2 = BinomialTree(random.random(), random.random(), random.random(), random.random(), 
                         random.random(), int(random.random()), typ='call', contract='American')
        self.assertEqual(bt2.contract, 'American')


    def test_bt_typ_invalid(self):
        with self.assertRaises(ValueError) as cm:
            BinomialTree(random.random(), random.random(), random.random(), random.random(), 
                         random.random(), int(random.random()), typ='Boo')
        self.assertIn('typ must be either', str(cm.exception))

    def test_bt_typ_valid(self):
        bt1 = BinomialTree(random.random(), random.random(), random.random(), random.random(), 
                         random.random(), int(random.random()), typ='call')
        self.assertEqual(bt1.typ, 'call')

        bt2 = BinomialTree(random.random(), random.random(), random.random(), random.random(), 
                         random.random(), int(random.random()), typ='put')
        self.assertEqual(bt2.typ, 'put')

    def test_bt_option_exp_call(self):
        bt = BinomialTree(100, random.random(), random.random(), random.random(), 120, 
                          int(random.random()), typ='call')
        self.assertEqual(bt.option_exp(120), 20)
        self.assertEqual(bt.option_exp(100), 0)
        self.assertEqual(bt.option_exp(90), 0)

    def test_bt_option_exp_put(self):
        bt = BinomialTree(100, random.random(), random.random(), random.random(), 120, 
                          int(random.random()), typ='put')
        self.assertEqual(bt.option_exp(120), 0)
        self.assertEqual(bt.option_exp(100), 0)
        self.assertEqual(bt.option_exp(90), 10)

    def test_bt_calc_delt(self):
        bt = BinomialTree(random.random(), random.random(), 1.1, 1/1.1, random.random(),
                           int(random.random()), typ='call')
        self.assertEqual(round(bt.calc_delta(120, 90, 100), 7), round(11/7, 7))
        self.assertEqual(bt.calc_delta(0, 0, 100), 0)

    def test_option_val(self):
        bt1 = BinomialTree(E=random.random(), mu=0, sigma=0.3, S_0=100, steps=100,
                          typ='call')
        self.assertEqual(round(bt1.option_val(0.05, 120, 100), 5), 109.96170) # Tests the functionn
        self.assertEqual(round(bt1.option_val(0, 120, 100), 5), 109.52381) # Tests the time value

    def test_stock_price(self):
        bt1 = BinomialTree(random.random(), T=1, steps=1, u=1.1, d=1/1.1, S_0=100,
                          typ='call')
        self.assertEqual(bt1.stock_price(), [[100], [round(100/1.1, 4), 110]])
        bt2 = BinomialTree(random.random(), T=1, steps=2, u=1.1, d=1/1.1, S_0=100,
                          typ='call')
        self.assertEqual(bt2.stock_price(), [[100], [round(100/1.1, 4), 110], [round(100/(1.1**2), 4), 100, round(100*(1.1**2), 4)]])
        #self.assertEqual(bt2.stock_price(flag=True), [[100], [round(100/1.1, 4), 110], [round(100/(1.1**2), 4), 100, round(100*(1.1**2), 4)]])
        
    def test_backprop(self):
        bt1 = BinomialTree(95, 1, 1.1, 1/1.1, 100, 1, 'call')
        self.assertEqual(round(bt1.backprop(r=0), 2), 7.14)
        
        bt2 = BinomialTree(95, 1, 1.1, 1/1.1, 100, 1, 'put')
        self.assertEqual(round(bt2.backprop(r=0), 2), 2.14)
        
        bt3 = BinomialTree(95, 1, 1.1, 1/1.1, 100, 1, typ='call')
        self.assertEqual(round(bt3.backprop(r=0.05), 2), 10.63)

        bt4 = BinomialTree(95, 1, 1.1, 1/1.1, 100, 1, typ='put')
        self.assertEqual(round(bt4.backprop(r=0.05), 2), 0.99)



if __name__ == "__main__":
    unittest.main()
