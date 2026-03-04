import unittest

import operator

from lore_sa.rule import Expression
from lore_sa.surrogate import DecisionTreeSurrogate


class RuleTest(unittest.TestCase):
    pass


class ExpressionTest(unittest.TestCase):

    def testExpression(self):
        exp = Expression("test", operator.gt,10)
        self.assertEqual(str(exp),"test > 10")

    def test_compact_premises(self):
        emitter = DecisionTreeSurrogate()
        premises = [Expression("class 1", operator.gt,10), Expression("class 2", operator.le,10), Expression("class 1", operator.gt,5)]
        compact = emitter.compact_premises(premises)
        self.assertEqual(str(compact[0]), str(Expression("class 1", operator.gt,10)))
        self.assertEqual(str(compact[1]), str(Expression("class 2", operator.le,10)))

if __name__ == '__main__':
    unittest.main()
