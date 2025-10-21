.. lore_Sa documentation master file, created by
   sphinx-quickstart on Tue Mar 21 12:03:48 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lore_sa - Local Rule-based Explanations
========================================

LORE (LOcal Rule-based Explanations) is a model-agnostic explanation method for black box 
classifiers. It provides interpretable explanations through decision rules, counterfactual 
scenarios, and feature importance scores.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: http://img.shields.io/badge/arXiv-1805.10820-B31B1B.svg
   :target: https://arxiv.org/abs/1805.10820
   :alt: arXiv Paper

**Key Features:**

* Model-agnostic: Works with any black box classifier
* Rule-based explanations: Natural IF-THEN rules
* Counterfactual reasoning: Shows "what-if" scenarios
* Feature importance: Identifies key decision factors
* Production-ready: Stable and actionable implementation

**Quick Links:**

* Paper: https://arxiv.org/abs/1805.10820
* GitHub: https://github.com/kdd-lab/LORE_sa
* Issues: https://github.com/kdd-lab/LORE_sa/issues

Getting Started
===============

.. toctree::
   :maxdepth: 2

   Get started <get_started>
   Architecture and Methodology <architecture>
   Tabular explanation example <examples/tabular_explanations_example>

API Reference
=============

**Modules**

.. toctree::
   :maxdepth: 2

   source/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
