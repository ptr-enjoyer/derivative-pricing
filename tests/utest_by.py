import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from binomial_tree.bt import BinomialTree
from binomial_tree.bt import Option
import unittest

class Test_Option(unittest.TestCase):

    def test_contract_invalid(self):
        with self.assertRaises(ValueError) as cm:
            Option(10, 10, 'call', contract='Foo')
        self.assertIn('contract must be either', str(cm.exception))

    def test_contract_valid(self):
        o1 = Option(10, 10, 'call', contract='European')
        self.assertEqual(o1.contract, 'European')

        o1 = Option(10, 10, 'call', contract='American')
        self.assertEqual(o1.contract, 'American')


if __name__ == "__main__":
    unittest.main()
