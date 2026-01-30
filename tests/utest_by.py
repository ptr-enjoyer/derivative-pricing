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










if __name__ == "__main__":
    unittest.main()
