import json

from .util import vector2dict
from typing import Callable
import operator

def json2expression(obj):
    return Expression(obj['att'], obj['op'], obj['thr'])


def json2rule(obj):
    premises = [json2expression(p) for p in obj['premise']]
    cons = obj['cons']
    return Rule(premises, cons)


class Expression(object):
    """
    Utility object to define a logical expression. It is used to define the premises of a Rule emitted from a surrogate model.
    """

    def __init__(self, variable: str, operator: Callable, value):
        """
        :param[str] variable: name of the variable that defines the rule
        :param[Callable] operator: logical operator involved in the rule
        :param value: numerical value to define the rule. E.g. variable > value 
        """

        self.variable = variable
        self.operator = operator
        self.value = value

    def operator2string(self):
        """
        it converts the logical operator into a string representation. E.g.: operator2string(operator.gt) = ">")
        """

        operator_strings = {operator.gt: '>', operator.lt: '<', operator.ne: '!=',
                            operator.eq: '=', operator.ge: '>=', operator.le: '<='}
        if self.operator not in operator_strings:
            raise ValueError(
                "logical operator not recognized. Use one of [operator.gt,operator.lt,operator.eq, operator.gte, operator.lte]")
        return operator_strings[self.operator]

    def __str__(self):
        """
        It writes the expression as a string
        """

        return "%s %s %s" % (self.variable, self.operator2string(), self.value)

    def __eq__(self, other):
        return (self.variable == other.variable and
                self.operator == other.operator and
                abs(self.value - other.value) < 1e-6)

    def to_dict(self):
        return {
            'att': self.variable,
            'op': self.operator2string(),
            'thr': self.value
        }


class Rule(object):

    def __init__(self, premises: list, consequences: Expression):
        """
        :param [list] premises: list of Expression objects representing the premises
        :param [Expression] consequences: Expression representing the consequence
        """
        # self.premises = [self.decode_rule(p) for p in premises]
        # self.consequences = self.decode_rule(consequences)
        self.premises = [p for p in premises]
        self.consequences = consequences

    def _pstr(self):
        return '{ %s }' % (', '.join([str(p) for p in self.premises]))

    def _cstr(self):
        return '{ %s }' % self.consequences

    def __str__(self):
        str_out = 'premises:\n' + '%s \n' % ("\n".join([str(e) for e in self.premises]))
        str_out += 'consequence: %s' % (str(self.consequences))

        return str_out

    def __eq__(self, other):
        return self.premises == other.premises and self.consequences == other.cons

    def __len__(self):
        return len(self.premises)

    def __hash__(self):
        return hash(str(self))

    def to_dict(self):
        premises = [{'attr': e.variable, 'val': e.value, 'op': e.operator2string()}
                    for e in self.premises]

        return {
            'premises': premises,
            'consequence': {
                'attr': self.consequences.variable,
                'val': self.consequences.value,
                'op': self.consequences.operator2string()
            }
        }

    def is_covered(self, x, feature_names):
        xd = vector2dict(x, feature_names)
        for p in self.premises:
            if p.operator == operator.le and xd[p.variable] > p.value:
                return False
            elif p.operator == operator.gt and xd[p.variable] <= p.value:
                return False
        return True


class ExpressionEncoder(json.JSONEncoder):
    """ Special json encoder for Condition types """

    def default(self, obj):
        if isinstance(obj, Expression):
            json_obj = {
                'att': obj.variable,
                'op': obj.operator2string(),
                'thr': obj.value,
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)


class RuleEncoder(json.JSONEncoder):
    """ Special json encoder for Rule types """

    def default(self, obj):
        if isinstance(obj, Rule):
            ce = ExpressionEncoder()
            json_obj = {
                'premise': [ce.default(p) for p in obj.premises],
                'cons': obj.consequences,
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)
