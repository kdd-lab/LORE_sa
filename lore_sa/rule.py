import json

from .encoder_decoder import EncDec
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
    Logical expression representing a condition in a rule.
    
    An Expression represents a single condition (premise) in a decision rule,
    consisting of a variable name, a comparison operator, and a threshold value.
    For example: "age > 30" or "income <= 50000".
    
    Expressions are used to build Rule objects that explain black box predictions.
    
    Attributes:
        variable (str): Name of the feature/variable in the condition
        operator (Callable): Comparison operator from the operator module 
            (e.g., operator.gt, operator.lt, operator.eq, operator.ge, operator.le)
        value (float or int): Threshold value for the comparison
    
    Example:
        >>> import operator
        >>> from lore_sa.rule import Expression
        >>> 
        >>> # Create an expression: age > 30
        >>> expr = Expression('age', operator.gt, 30)
        >>> print(expr)  # Output: age > 30
        >>> 
        >>> # Create an expression: income <= 50000
        >>> expr2 = Expression('income', operator.le, 50000)
        >>> print(expr2)  # Output: income <= 50000
    """

    def __init__(self, variable: str, operator: Callable, value):
        """
        Initialize a logical expression.
        
        Args:
            variable (str): Name of the variable/feature that the expression refers to
            operator (Callable): Logical comparison operator. Use one from the operator 
                module: operator.gt (>), operator.lt (<), operator.eq (=), 
                operator.ge (>=), operator.le (<=), operator.ne (!=)
            value (float or int): Numerical threshold value to compare against
        
        Raises:
            ValueError: If an unsupported operator is provided
        """

        self.variable = variable
        self.operator = operator
        self.value = value

    def operator2string(self):
        """
        Convert the logical operator to its string representation.
        
        Returns:
            str: String representation of the operator (e.g., '>', '<', '=', '>=', '<=', '!=')
        
        Raises:
            ValueError: If the operator is not one of the recognized comparison operators
        
        Example:
            >>> import operator
            >>> expr = Expression('age', operator.gt, 30)
            >>> expr.operator2string()  # Returns '>'
        """

        operator_strings = {operator.gt: '>', operator.lt: '<', operator.ne: '!=',
                            operator.eq: '=', operator.ge: '>=', operator.le: '<='}
        if self.operator not in operator_strings:
            raise ValueError(
                "logical operator not recognized. Use one of [operator.gt,operator.lt,operator.eq, operator.gte, operator.lte]")
        return operator_strings[self.operator]

    def __str__(self):
        """
        Return a human-readable string representation of the expression.
        
        Returns:
            str: String in the format "variable operator value" (e.g., "age > 30")
        """

        return "%s %s %s" % (self.variable, self.operator2string(), self.value)

    def __eq__(self, other):
        return (self.variable == other.variable and
                self.operator == other.operator and
                abs(self.value - other.value) < 1e-6)

    def to_dict(self):
        """
        Convert the expression to a dictionary representation.
        
        Returns:
            dict: Dictionary with keys 'att' (attribute/variable name), 
                'op' (operator as string), and 'thr' (threshold value)
        
        Example:
            >>> expr = Expression('age', operator.gt, 30)
            >>> expr.to_dict()
            {'att': 'age', 'op': '>', 'thr': 30}
        """
        return {
            'att': self.variable,
            'op': self.operator2string(),
            'thr': self.value
        }


class Rule(object):
    """
    Decision rule with premises (conditions) and consequences (predictions).
    
    A Rule represents an if-then decision rule extracted from an interpretable model.
    It consists of:
    - Premises: A list of Expression objects that form the "if" part (conditions)
    - Consequences: An Expression representing the "then" part (predicted class)
    
    Rules are the primary output of LORE explanations, describing when and why 
    a black box model makes specific predictions.
    
    Attributes:
        premises (list): List of Expression objects representing the conditions
        consequences (Expression): Expression representing the predicted outcome
        encoder (EncDec): Encoder/decoder for handling feature transformations
    
    Example:
        >>> # Rule: IF age > 30 AND income <= 50000 THEN class = 0
        >>> premises = [
        ...     Expression('age', operator.gt, 30),
        ...     Expression('income', operator.le, 50000)
        ... ]
        >>> consequence = Expression('class', operator.eq, 0)
        >>> rule = Rule(premises, consequence, encoder)
        >>> print(rule)  # Output: { age > 30, income <= 50000 } --> { class = 0 }
    """

    def __init__(self, premises: list, consequences: Expression, encoder: EncDec):
        """
        Initialize a decision rule.
        
        Args:
            premises (list): List of Expression objects representing the rule's conditions.
                These are combined with AND logic.
            consequences (Expression): Expression representing the rule's prediction/outcome
            encoder (EncDec): Encoder/decoder object used to decode categorical features 
                back to their original representation
        
        Note:
            The encoder is used to decode one-hot encoded categorical features back to 
            their original categorical values, making the rule more interpretable.
        """
        self.encoder = encoder
        self.premises = [self.decode_rule(p) for p in premises]
        self.consequences = self.decode_rule(consequences)

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


    def decode_rule(self, rule: Expression):
        if 'categorical' not in self.encoder.dataset_descriptor.keys() or self.encoder.dataset_descriptor['categorical'] == {}:
            return rule

        if rule.variable.split('=')[0] in self.encoder.dataset_descriptor['categorical'].keys():
            decoded_label = rule.variable.split("=")[0]
            decoded_value = rule.variable.split("=")[1]
            rule.variable = decoded_label
            if rule.value:
                rule.operator = operator.eq
            else:
                rule.operator = operator.ne
            rule.value = decoded_value
            return rule
        else:
            return rule

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
