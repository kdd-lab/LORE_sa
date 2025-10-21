
# LOcal Rule-based Explanation
![Tests](https://github.com/kdd-lab/LORE_sa/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![IEEE](https://img.shields.io/badge/IEEE-8920138-00629B.svg)](https://ieeexplore.ieee.org/document/8920138)
[![arXiv](http://img.shields.io/badge/arXiv-1805.10820-B31B1B.svg)](https://arxiv.org/abs/1805.10820)
[![GitHub contributors](https://img.shields.io/github/contributors/kdd-lab/LORE_sa)](https://github.com/kdd-lab/LORE_sa/contributors)

**Stable & Actionable**

Official repository of the LORE (Local Rule-Based Explanation) algorithm.

## Overview

LORE is a model-agnostic explanation method that provides **interpretable explanations** for black box classifier predictions. It generates explanations in the form of:

- **Decision rules**: IF-THEN statements explaining why a prediction was made
- **Counterfactual rules**: "What-if" scenarios showing what changes would lead to different predictions
- **Feature importance**: Scores indicating which features were most relevant to the decision

### Key Features

✅ **Model-agnostic**: Works with any black box classifier (scikit-learn, Keras, PyTorch, etc.)  
✅ **Human-interpretable**: Provides natural language-like IF-THEN rules  
✅ **Counterfactual reasoning**: Shows minimal changes needed for different predictions  
✅ **Local explanations**: Explains individual predictions with high fidelity  
✅ **Production-ready**: Stable implementation suitable for real-world applications  

### How LORE Works

LORE explains individual predictions through a four-stage process:

1. **Encoding**: Transform the instance to an encoded representation
2. **Neighborhood Generation**: Create synthetic instances around the instance to explain using genetic algorithms or random sampling
3. **Surrogate Training**: Train an interpretable decision tree on the neighborhood labeled by the black box
4. **Rule Extraction**: Extract factual and counterfactual rules from the surrogate model

For detailed methodology, see the paper:

> Guidotti, R., Monreale, A., Ruggieri, S., Pedreschi, D., Turini, F., & Giannotti, F. (2018).  
> Local rule-based explanations of black box decision systems.  
> arXiv:1805.10820. https://arxiv.org/abs/1805.10820 


## Getting started

### Installation

We suggest to install the library and its requirements into a dedicated environment.

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt 
```

### Quick Example

To use the library within your project, import the needed packages:

```python
from lore_sa import TabularGeneticGeneratorLore
from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearn_classifier_bbox

# 1. Load your dataset
dataset = TabularDataset.from_csv('my_data.csv', class_name="class")

# 2. Wrap your trained model
bbox = sklearn_classifier_bbox.sklearnBBox(trained_model)

# 3. Create the LORE explainer
explainer = TabularGeneticGeneratorLore(bbox, dataset)

# 4. Explain a single instance
explanation = explainer.explain_instance(instance)

# 5. Access explanation components
print("Factual rule:", explanation['rule'])
print("Counterfactuals:", explanation['counterfactuals'])
print("Feature importances:", explanation['feature_importances'])
print("Fidelity:", explanation['fidelity'])
```
## Complete Example

Let's walk through a complete example explaining a Random Forest classifier on a credit risk dataset:

### Step 1: Prepare the Model

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lore_sa.bbox import sklearn_classifier_bbox

# Load and split data
df = pd.read_csv('data/credit_risk.csv')
X = df.drop('class', axis=1).values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train your black box model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Wrap the model
bbox = sklearn_classifier_bbox.sklearnBBox(model)
```

### Step 2: Create the Dataset

```python
from lore_sa.dataset import TabularDataset

# Create dataset with feature information
dataset = TabularDataset.from_csv('data/credit_risk.csv', class_name='class')

# Optional: specify categorical and ordinal columns explicitly
dataset.update_descriptor(
    categorial_columns=['workclass', 'education', 'marital-status', 'occupation'],
    ordinal_columns=['education-level']
)
```

### Step 3: Create and Use the Explainer

```python
from lore_sa import TabularGeneticGeneratorLore

# Create LORE explainer
explainer = TabularGeneticGeneratorLore(bbox, dataset)

# Explain a single instance
instance = X_test[0]
explanation = explainer.explain_instance(instance)

# Print the explanation
print("\n=== LORE Explanation ===")
print(f"\nFactual Rule: {explanation['rule']}")
print(f"\nFidelity: {explanation['fidelity']:.2f}")
print(f"\nTop 5 Features:")
for feature, importance in explanation['feature_importances'][:5]:
    print(f"  - {feature}: {importance:.3f}")

print(f"\nCounterfactuals ({len(explanation['counterfactuals'])} found):")
for i, cf in enumerate(explanation['counterfactuals'][:3], 1):
    print(f"  {i}. {cf}")
```

### Understanding the Output

**Factual Rule**: Explains the current prediction
```
IF age > 30 AND income <= 50000 AND education = 'Bachelor' 
THEN prediction = 'denied'
```

**Counterfactual Rules**: Show alternative scenarios
```
IF income > 50000 THEN prediction = 'approved'
```

**Deltas**: Minimal changes needed
```
Changes needed: [income: 45000 → >50000]
```

**Fidelity**: Reliability of the explanation (0.95 = 95% agreement with black box)

## Choosing an Explainer

LORE provides three pre-configured explainer variants:

### TabularGeneticGeneratorLore (Recommended)

Uses a genetic algorithm to generate high-quality neighborhoods. Best for most use cases.

```python
from lore_sa import TabularGeneticGeneratorLore
explainer = TabularGeneticGeneratorLore(bbox, dataset)
```

**Pros**: High-quality explanations, good fidelity  
**Cons**: Slower than random generation  
**Best for**: Production use, complex models, when explanation quality is critical

### TabularRandomGeneratorLore

Uses random sampling for neighborhood generation. Fastest but may produce less accurate explanations.

```python
from lore_sa import TabularRandomGeneratorLore
explainer = TabularRandomGeneratorLore(bbox, dataset)
```

**Pros**: Very fast  
**Cons**: Lower fidelity, may miss important patterns  
**Best for**: Quick exploratory analysis, simple models

### TabularRandGenGeneratorLore

Probabilistic variant combining genetic and random approaches.

```python
from lore_sa import TabularRandGenGeneratorLore
explainer = TabularRandGenGeneratorLore(bbox, dataset)
```

**Pros**: Balance of speed and quality  
**Cons**: Not as thorough as pure genetic  
**Best for**: Medium-complexity models, time constraints

## Advanced Usage

### Custom Configuration

For more control, you can configure LORE components manually:

```python
from lore_sa.lore import Lore
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.neighgen import GeneticGenerator
from lore_sa.surrogate import DecisionTreeSurrogate

# Create components
encoder = ColumnTransformerEnc(dataset.descriptor)
generator = GeneticGenerator(bbox, dataset, encoder, ocr=0.1)
surrogate = DecisionTreeSurrogate(prune_tree=True)

# Create explainer
explainer = Lore(bbox, dataset, encoder, generator, surrogate)

# Generate explanation with custom parameters
explanation = explainer.explain(instance, num_instances=1500)
```

### Working with Different Data Types

LORE automatically handles:
- **Numerical features**: Continuous or discrete values
- **Categorical features**: Nominal categories (one-hot encoded internally)
- **Ordinal features**: Ordered categories

Specify feature types when creating the dataset:

```python
dataset = TabularDataset.from_csv(
    'data.csv',
    class_name='target',
    categorial_columns=['color', 'size', 'type'],
    ordinal_columns=['quality_level']
)
```

## Documentation

Comprehensive documentation is available at: https://kdd-lab.github.io/LORE_sa/html/index.html

### Key Documentation Pages

- **[Get Started](https://kdd-lab.github.io/LORE_sa/html/get_started.html)**: Installation and quick start guide
- **[Architecture](https://kdd-lab.github.io/LORE_sa/html/architecture.html)**: Detailed methodology and components
- **[API Reference](https://kdd-lab.github.io/LORE_sa/html/source/modules.html)**: Complete API documentation
- **[Examples](https://kdd-lab.github.io/LORE_sa/html/examples/tabular_explanations_example.html)**: Full tutorial notebooks

### Building Documentation Locally

The documentation is based on Sphinx. To build it locally:

```bash
cd docs
make html
```

Once built, the documentation is available in `docs/_build/html/index.html`.

### Updating Online Documentation

To update the online documentation:

1. Build the documentation: `cd docs && make html`
2. Copy the build: `rm -rf docs/html && cp -r docs/_build/html docs/html`
3. Commit and push: The documentation is automatically published via GitHub Pages




## Contributing

We welcome contributions to LORE_sa! Here's how you can help:

### Reporting Issues

For bugs or feature requests, please open an issue at: https://github.com/kdd-lab/LORE_sa/issues

When reporting a bug, please include:
- Python version
- Library versions (from `pip freeze`)
- Minimal code to reproduce the issue
- Expected vs actual behavior

### Contributing Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest test/`
5. Commit your changes: `git commit -m "Add my feature"`
6. Push to your fork: `git push origin feature/my-feature`
7. Open a pull request

**Requirements for PR acceptance:**
- All tests must pass
- Code must follow existing style conventions
- New features should include tests
- Documentation should be updated if needed

## Citation

If you use LORE in your research, please cite:

```bibtex
@article{guidotti2018local,
  title={Local rule-based explanations of black box decision systems},
  author={Guidotti, Riccardo and Monreale, Anna and Ruggieri, Salvatore and 
          Pedreschi, Dino and Turini, Franco and Giannotti, Fosca},
  journal={arXiv preprint arXiv:1805.10820},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Paper authors: Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Dino Pedreschi, Franco Turini, Fosca Giannotti
- Contributors: See [CONTRIBUTORS](https://github.com/kdd-lab/LORE_sa/contributors)

## Related Projects

- **LIME** (Local Interpretable Model-agnostic Explanations): Uses linear models for local explanations
- **SHAP** (SHapley Additive exPlanations): Uses Shapley values for feature attribution
- **Anchor**: Provides high-precision rules for explanations
- **DiCE** (Diverse Counterfactual Explanations): Focuses on generating diverse counterfactuals

## Support

- 📖 Documentation: https://kdd-lab.github.io/LORE_sa/html/index.html
- 🐛 Issue Tracker: https://github.com/kdd-lab/LORE_sa/issues
- 📄 Paper: https://arxiv.org/abs/1805.10820
- 💬 Discussions: Use GitHub Discussions for questions and discussions

