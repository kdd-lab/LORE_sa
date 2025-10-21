====================================
Architecture and Methodology
====================================

Overview
========

LORE (LOcal Rule-based Explanations) is a model-agnostic explanation framework designed to provide 
interpretable explanations for black box classifier predictions. This document describes the 
architecture, methodology, and theoretical foundation of LORE.

Theoretical Foundation
======================

LORE is based on the following principles:

1. **Local Approximation**: Rather than explaining the entire model, LORE explains individual predictions by creating a local approximation around the instance of interest.

2. **Rule-Based Explanations**: Uses decision rules (IF-THEN statements) that are naturally interpretable to humans.

3. **Counterfactual Reasoning**: Provides "what-if" scenarios showing what changes would lead to different predictions.

4. **Model Agnostic**: Works with any black box classifier that provides a prediction interface.

The Four-Stage Process
=======================

LORE follows a four-stage process to generate explanations:

Stage 1: Instance Encoding
---------------------------

The instance to explain is transformed from its original feature space to an encoded space suitable 
for machine learning operations.

**Purpose:**
- Handle categorical features through one-hot encoding
- Normalize numerical features if needed
- Create a consistent representation for the neighborhood generation

**Components:**
- ``EncDec``: Abstract encoder/decoder interface
- ``ColumnTransformerEnc``: Concrete implementation for tabular data

**Example:**

.. code-block:: python

    # Original instance: ['red', 30, 50000]
    # Encoded instance:  [1, 0, 0, 30, 50000]  (one-hot for color)
    
    encoded_instance = encoder.encode([instance])

Stage 2: Neighborhood Generation
---------------------------------

A synthetic neighborhood of instances is generated around the encoded instance. This neighborhood 
serves as training data for the surrogate model.

**Purpose:**
- Explore the local decision boundary
- Generate instances with diverse predicted classes
- Create a representative sample of the local feature space

**Strategies:**

1. **Random Generation** (``RandomGenerator``):
   
   - Samples feature values uniformly from valid ranges
   - Fast but may miss important decision boundaries
   - Best for simple, well-behaved decision spaces

2. **Genetic Algorithm** (``GeneticGenerator``):
   
   - Evolves a population of instances using genetic operators
   - Optimizes for: (1) similarity to original instance, (2) diversity in predictions
   - Provides better coverage of decision boundaries
   - Recommended for most use cases

3. **Probabilistic Genetic** (``GeneticProbaGenerator``):
   
   - Adds probabilistic elements to genetic evolution
   - Balance between diversity and computational cost

**Key Parameters:**

- ``num_instances``: Number of synthetic instances (default: 1000)
- ``ocr`` (One-Class Ratio): Balance between same-class and different-class instances (default: 0.1)

**Example:**

.. code-block:: python

    # Generate 1000 synthetic instances around the encoded instance
    neighborhood = generator.generate(
        encoded_instance, 
        num_instances=1000,
        descriptor=dataset.descriptor,
        encoder=encoder
    )

Stage 3: Surrogate Training
----------------------------

An interpretable surrogate model (typically a decision tree) is trained on the neighborhood, 
using predictions from the black box as labels.

**Purpose:**
- Create a local approximation of the black box
- Provide an interpretable structure for rule extraction
- Measure fidelity (agreement with black box)

**Process:**

1. Generate predictions for all neighborhood instances using the black box
2. Train a decision tree on (neighborhood, predictions) pairs
3. Optionally prune the tree to improve interpretability
4. Compute fidelity score

**Fidelity:**

Fidelity measures how well the surrogate approximates the black box in the local neighborhood:

.. math::

    fidelity = \\frac{\\text{agreements}}{\\text{total instances}}

A fidelity score close to 1.0 indicates high reliability of the explanation.

**Example:**

.. code-block:: python

    # Get black box predictions for neighborhood
    bbox_predictions = bbox.predict(decoded_neighborhood)
    
    # Train decision tree surrogate
    surrogate.train(neighborhood, bbox_predictions)
    
    # Check fidelity
    print(f"Fidelity: {surrogate.fidelity:.2f}")

Stage 4: Rule Extraction
-------------------------

Decision rules are extracted from the trained surrogate to provide the explanation.

**Factual Rule:**

Describes the path in the decision tree that the instance follows, explaining why the 
black box made its prediction.

.. code-block:: text

    IF age > 30 AND income <= 50000 AND education = 'Bachelor' 
    THEN prediction = 'denied'

**Counterfactual Rules:**

Describe alternative paths in the tree leading to different predictions, showing what 
changes would alter the outcome.

.. code-block:: text

    IF age > 30 AND income > 50000 
    THEN prediction = 'approved'

**Deltas (Minimal Changes):**

For each counterfactual, LORE computes the minimal set of feature changes needed:

.. code-block:: text

    To change from 'denied' to 'approved':
    - Change: income from 45000 to > 50000
    - Keep: age = 35 (unchanged)
    - Keep: education = 'Bachelor' (unchanged)

Architecture Components
=======================

Black Box Wrapper (AbstractBBox)
---------------------------------

**Purpose:** Provides a consistent interface to any machine learning model.

**Requirements:**
- ``predict(X)``: Returns class predictions
- ``predict_proba(X)``: Returns class probabilities

**Implementations:**
- ``sklearnBBox``: For scikit-learn models
- ``KerasClassifierWrapper``: For Keras/TensorFlow models
- Custom wrappers can be created by inheriting from ``AbstractBBox``

**Example:**

.. code-block:: python

    from lore_sa.bbox import sklearn_classifier_bbox
    
    # Wrap a scikit-learn pipeline
    bbox = sklearn_classifier_bbox.sklearnBBox(sklearn_pipeline)
    
    # Now it has a consistent interface
    predictions = bbox.predict(X)
    probabilities = bbox.predict_proba(X)

Dataset (TabularDataset)
-------------------------

**Purpose:** Stores data and metadata about features (types, ranges, categories).

**Descriptor Structure:**

.. code-block:: python

    descriptor = {
        'numeric': {
            'age': {
                'index': 0,
                'min': 18,
                'max': 90,
                'mean': 45.2,
                'std': 12.5,
                'median': 44,
                'q1': 35,
                'q3': 55
            },
            'income': { ... }
        },
        'categorical': {
            'education': {
                'index': 2,
                'distinct_values': ['High School', 'Bachelor', 'Master', 'PhD'],
                'value_counts': {'Bachelor': 450, 'Master': 300, ...}
            }
        },
        'ordinal': { ... }
    }

**Methods:**

- ``from_csv()``: Load dataset from CSV file
- ``update_descriptor()``: Recompute feature statistics
- Access via ``dataset.df`` (pandas DataFrame)

Encoder/Decoder (EncDec)
-------------------------

**Purpose:** Transform features between original and encoded spaces.

**Key Operations:**

1. **Encoding**: Original → Encoded
   
   - One-hot encode categorical features
   - Keep numerical features as-is (or scale if needed)
   
2. **Decoding**: Encoded → Original
   
   - Reverse one-hot encoding for categorical features
   - Map back to original feature space

3. **Feature Mapping**:
   
   - Track which encoded indices correspond to which original features
   - Maintain intervals for grouped features (e.g., one-hot encoded categories)

**Example:**

.. code-block:: python

    # Original: [30, 'red', 50000]
    encoded = encoder.encode([[30, 'red', 50000]])
    # Result: [[30, 1, 0, 0, 50000]]  (one-hot for 'red')
    
    decoded = encoder.decode(encoded)
    # Result: [[30, 'red', 50000]]  (back to original)
    
    # Get feature mapping
    features = encoder.get_encoded_features()
    # {0: 'age', 1: 'color=red', 2: 'color=blue', 
    #  3: 'color=green', 4: 'income'}

Neighborhood Generator (NeighborhoodGenerator)
-----------------------------------------------

**Purpose:** Generate synthetic instances for surrogate training.

**Parameters:**

- ``bbox``: Black box to query for predictions
- ``dataset``: Dataset with feature descriptors
- ``encoder``: Encoder for feature transformations
- ``ocr``: One-Class Ratio (balance between same/different class instances)

**Genetic Algorithm Details:**

For ``GeneticGenerator``, the genetic algorithm uses:

- **Fitness Functions**:
  
  - For same-class instances: Maximize similarity to original + same class prediction
  - For different-class instances: Maximize similarity + different class prediction

- **Genetic Operators**:
  
  - **Crossover**: Combine features from two parent instances
  - **Mutation**: Randomly change feature values
  - **Selection**: Tournament selection based on fitness

- **Parameters**:
  
  - ``ngen``: Number of generations (default: 100)
  - ``mutpb``: Mutation probability (default: 0.2)
  - ``cxpb``: Crossover probability (default: 0.5)
  - ``alpha1``, ``alpha2``: Weights for similarity vs. diversity (default: 0.5, 0.5)

Surrogate Model (Surrogate)
----------------------------

**Purpose:** Interpretable model that approximates black box locally.

**Decision Tree Surrogate:**

The ``DecisionTreeSurrogate`` uses scikit-learn's DecisionTreeClassifier with:

- **Class balancing**: ``class_weight='balanced'`` to handle imbalanced neighborhoods
- **Optional pruning**: Grid search to find optimal tree complexity
- **Random state**: Fixed for reproducibility

**Pruning Parameters** (when enabled):

.. code-block:: python

    param_list = {
        'min_samples_split': [0.01, 0.05, 0.1, 0.2, 3, 2],
        'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 2, 4],
        'splitter': ['best', 'random'],
        'max_depth': [None, 2, 10, 12, 16, 20, 30],
        'criterion': ['entropy', 'gini'],
        'max_features': [0.2, 1, 5, 'auto', 'sqrt', 'log2']
    }

Rule and Expression Classes
----------------------------

**Expression**: Single condition in a rule

.. code-block:: python

    # Represents: age > 30
    expr = Expression('age', operator.gt, 30)

**Rule**: Complete IF-THEN statement

.. code-block:: python

    # IF age > 30 AND income <= 50000 THEN class = 0
    premises = [
        Expression('age', operator.gt, 30),
        Expression('income', operator.le, 50000)
    ]
    consequence = Expression('class', operator.eq, 0)
    rule = Rule(premises, consequence, encoder)

Workflow Diagram
================

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────┐
    │ 1. Instance to Explain                                      │
    │    [age=35, color='red', income=45000]                     │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 2. Encoding                                                 │
    │    [35, 1, 0, 0, 45000]  (one-hot for color)              │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 3. Neighborhood Generation                                  │
    │    1000 synthetic instances around encoded instance         │
    │    - 900 with same predicted class                         │
    │    - 100 with different predicted classes                  │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 4. Black Box Labeling                                       │
    │    Get predictions for all neighborhood instances           │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 5. Surrogate Training                                       │
    │    Train decision tree on (neighborhood, predictions)       │
    │    Compute fidelity score                                   │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 6. Rule Extraction                                          │
    │    • Factual: Why this prediction?                         │
    │    • Counterfactuals: What if scenarios?                   │
    │    • Deltas: Minimal changes needed                        │
    │    • Feature Importances: Which features matter most?      │
    └─────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 7. Explanation Output                                       │
    │    Interpretable rules in original feature space           │
    └─────────────────────────────────────────────────────────────┘

Best Practices
==============

Choosing num_instances
-----------------------

- **Small datasets (<1000 instances)**: 500-1000 synthetic instances
- **Medium datasets (1000-10000)**: 1000-2000 synthetic instances  
- **Large datasets (>10000)**: 1500-3000 synthetic instances

More instances improve explanation quality but increase computation time.

Interpreting Fidelity
----------------------

- **fidelity > 0.9**: Excellent - explanation is highly reliable
- **0.7 < fidelity ≤ 0.9**: Good - explanation is generally reliable
- **0.5 < fidelity ≤ 0.7**: Fair - some uncertainty in explanation
- **fidelity ≤ 0.5**: Poor - explanation may not be reliable, increase num_instances

Handling Low Fidelity
----------------------

If fidelity is low, try:

1. Increase ``num_instances`` (e.g., from 1000 to 2000)
2. Use ``GeneticGenerator`` instead of ``RandomGenerator``
3. Enable tree pruning in ``DecisionTreeSurrogate``
4. Check if the instance is an outlier or near a complex decision boundary

Feature Importance Interpretation
----------------------------------

Feature importances are computed from the decision tree's feature importances, adjusted 
for one-hot encoded features. Higher values indicate greater importance in the decision.

**Note**: Importances are local to the specific instance and may differ from global 
feature importances.

Computational Complexity
========================

Time Complexity
---------------

For a single explanation:

- **Random Generation**: O(n × f) where n = num_instances, f = num_features
- **Genetic Generation**: O(g × p × f × t) where g = generations, p = population size, 
  f = num_features, t = tree depth
- **Surrogate Training**: O(n × f × log(n)) for decision tree

Space Complexity
----------------

- **Neighborhood**: O(n × f_enc) where f_enc = number of encoded features
- **Decision Tree**: O(nodes) where nodes depends on tree depth and complexity

Typical Running Times
---------------------

On a modern CPU (e.g., Intel i7):

- **RandomGenerator**: 1-2 seconds per instance
- **GeneticGenerator**: 5-15 seconds per instance
- **Explanation extraction**: <1 second

References
==========

Primary Paper
-------------

.. code-block:: text

    Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F., & Giannotti, F. (2018).
    Local rule-based explanations of black box decision systems.
    arXiv preprint arXiv:1805.10820.
    https://arxiv.org/abs/1805.10820

Related Work
------------

- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Counterfactual Explanations
- Rule-based Machine Learning

Key Differences from LIME
--------------------------

- **Rules vs Linear**: LORE uses rules (IF-THEN), LIME uses linear models
- **Counterfactuals**: LORE provides explicit counterfactual scenarios
- **Deltas**: LORE computes minimal changes needed for different predictions
- **Neighborhood**: LORE uses genetic algorithms for better neighborhoods

Implementation Notes
====================

Thread Safety
-------------

The LORE implementation is not thread-safe. Create separate explainer instances for 
concurrent explanations.

Memory Considerations
---------------------

Large neighborhoods can consume significant memory. For memory-constrained environments:

- Reduce ``num_instances``
- Process instances in batches
- Use ``RandomGenerator`` instead of ``GeneticGenerator``

Reproducibility
---------------

For reproducible explanations:

- Set ``random_seed`` parameter in generators
- Use fixed ``random_state`` in decision trees
- Ensure consistent data preprocessing
