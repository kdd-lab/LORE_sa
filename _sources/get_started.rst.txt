===========
Get Started
===========

Welcome to LORE_sa
==================

LORE (LOcal Rule-based Explanations) is a model-agnostic explanation method for black box classifiers. 
It provides interpretable explanations for individual predictions by generating decision rules, 
counterfactual scenarios, and feature importance scores.

This implementation is stable and actionable, designed for production use with tabular data.

What is LORE?
=============

LORE is an explanation method that answers three key questions about a black box model's prediction:

1. **Why?** - Provides a decision rule explaining the prediction
2. **What if?** - Shows counterfactual rules for different predictions
3. **Which features?** - Identifies the most important features

The method works by:

1. Generating a synthetic neighborhood around the instance to explain
2. Training an interpretable surrogate model (decision tree) on this neighborhood
3. Extracting rules and counterfactuals from the surrogate
4. Computing feature importances

For more details, see the paper:

    Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F., & Giannotti, F. (2018).
    Local rule-based explanations of black box decision systems. 
    arXiv:1805.10820. https://arxiv.org/abs/1805.10820

Installation
============

Prerequisites
-------------

- Python 3.7 or higher
- pip package manager

We recommend using a virtual environment to avoid dependency conflicts.

Using virtualenv
----------------

.. code-block:: bash

    # Create a virtual environment
    virtualenv venv
    
    # Activate the environment (Linux/Mac)
    source venv/bin/activate
    
    # Activate the environment (Windows)
    venv\Scripts\activate
    
    # Install requirements
    pip install -r requirements.txt

Using conda
-----------

.. code-block:: bash

    # Create a conda environment
    conda create -n lore_env python=3.9
    
    # Activate the environment
    conda activate lore_env
    
    # Install requirements
    pip install -r requirements.txt

Quick Start
===========

Basic Usage
-----------

Here's a minimal example to get you started:

.. code-block:: python

    from lore_sa import TabularGeneticGeneratorLore
    from lore_sa.dataset import TabularDataset
    from lore_sa.bbox import sklearn_classifier_bbox
    
    # 1. Load your dataset
    dataset = TabularDataset.from_csv('data.csv', class_name='target')
    
    # 2. Wrap your trained model
    bbox = sklearn_classifier_bbox.sklearnBBox(trained_model)
    
    # 3. Create the LORE explainer
    explainer = TabularGeneticGeneratorLore(bbox, dataset)
    
    # 4. Explain a single instance
    explanation = explainer.explain_instance(instance)
    
    # 5. Access the explanation components
    print("Factual rule:", explanation['rule'])
    print("Fidelity:", explanation['fidelity'])
    print("Top features:", explanation['feature_importances'][:5])

Key Components
==============

LORE consists of four main components:

1. **Black Box Wrapper** (``AbstractBBox``)
   
   Wraps your machine learning model to provide a consistent interface:
   
   - ``sklearn_classifier_bbox.sklearnBBox`` for scikit-learn models
   - ``keras_classifier_wrapper`` for Keras/TensorFlow models

2. **Dataset** (``TabularDataset``)
   
   Contains your data and feature descriptors (types, ranges, categories):
   
   .. code-block:: python
   
       dataset = TabularDataset.from_csv('data.csv', class_name='target')

3. **Encoder/Decoder** (``EncDec``)
   
   Handles feature transformations (e.g., one-hot encoding for categorical features):
   
   .. code-block:: python
   
       from lore_sa.encoder_decoder import ColumnTransformerEnc
       encoder = ColumnTransformerEnc(dataset.descriptor)

4. **Neighborhood Generator** (``NeighborhoodGenerator``)
   
   Creates synthetic instances around the instance to explain:
   
   - ``RandomGenerator``: Simple random sampling
   - ``GeneticGenerator``: Genetic algorithm for better neighborhoods
   - ``GeneticProbaGenerator``: Probabilistic genetic variant

Choosing an Explainer
======================

LORE provides three pre-configured explainer classes:

TabularRandomGeneratorLore
---------------------------

Uses random sampling for neighborhood generation. Fastest but may produce less accurate explanations.

.. code-block:: python

    from lore_sa import TabularRandomGeneratorLore
    explainer = TabularRandomGeneratorLore(bbox, dataset)

**Best for:** Quick exploratory analysis, simple datasets

TabularGeneticGeneratorLore
----------------------------

Uses a genetic algorithm to evolve high-quality neighborhoods. Recommended for most use cases.

.. code-block:: python

    from lore_sa import TabularGeneticGeneratorLore
    explainer = TabularGeneticGeneratorLore(bbox, dataset)

**Best for:** Production use, complex datasets, when explanation quality is critical

TabularRandGenGeneratorLore
----------------------------

Uses a probabilistic genetic algorithm. Balance between speed and quality.

.. code-block:: python

    from lore_sa import TabularRandGenGeneratorLore
    explainer = TabularRandGenGeneratorLore(bbox, dataset)

**Best for:** Medium-complexity datasets, when you need a balance of speed and quality

Understanding the Explanation
==============================

The ``explain_instance()`` method returns a dictionary with several components:

Rule
----

The factual rule explaining the prediction:

.. code-block:: python

    rule = explanation['rule']
    # Example: IF age > 30 AND income <= 50000 THEN class = 0

Counterfactuals
---------------

Alternative scenarios that would lead to different predictions:

.. code-block:: python

    counterfactuals = explanation['counterfactuals']
    # Example: IF age > 30 OR income > 50000 THEN class = 1

Deltas
------

Minimal changes needed to reach each counterfactual:

.. code-block:: python

    deltas = explanation['deltas']
    # Example: [income > 50000] (increase income to change prediction)

Feature Importances
-------------------

Importance of each feature in the decision:

.. code-block:: python

    importances = explanation['feature_importances']
    # Example: [('age', 0.45), ('income', 0.32), ('education', 0.15), ...]

Fidelity
--------

How well the explanation approximates the black box (0 to 1):

.. code-block:: python

    fidelity = explanation['fidelity']
    # Example: 0.95 (the surrogate agrees with the black box 95% of the samples in the neighborhood)

A fidelity close to 1.0 indicates the explanation is highly reliable.

Next Steps
==========

- See the :doc:`examples/tabular_explanations_example` for a complete walkthrough
- Check the :doc:`source/modules` for detailed API reference
- Read the paper at https://arxiv.org/abs/1805.10820 for the theoretical foundation

Common Pitfalls
===============

1. **Missing target class**: Ensure you specify ``class_name`` when creating the dataset
2. **Feature order**: The instance to explain must have features in the same order as the training data
3. **Low fidelity**: If fidelity is low (<0.7), try increasing ``num_instances`` in ``explain()``
4. **Categorical features**: Make sure categorical columns are properly identified in the dataset

Getting Help
============

- GitHub Issues: https://github.com/kdd-lab/LORE_sa/issues
- Documentation: https://kdd-lab.github.io/LORE_sa/html/index.html