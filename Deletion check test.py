import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from lore_sa import sklearn_classifier_bbox

# Load dataset
df = pd.read_csv('german_credit.csv')

# Define X and y
X = df.drop(columns='default')
y = df['default']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [1, 4, 7, 10, 12, 15, 17]),
        ('cat', OrdinalEncoder(), [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])
    ]
)

# Build model pipeline
model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Fit model
model.fit(X_train, y_train)

bbox = sklearn_classifier_bbox.sklearnBBox(model)

from lore_sa.dataset import TabularDataset
from lore_sa.neighgen import GeneticGenerator
from lore_sa.encoder_decoder import ColumnTransformerEnc
from lore_sa.lore import Lore
from lore_sa.surrogate import DecisionTreeSurrogate
from sklearn.preprocessing import FunctionTransformer  # For identity encoding

# Load dataset
dataset = TabularDataset.from_csv('german_credit.csv', class_name="default")
dataset.df.dropna(inplace=True)
dataset.update_descriptor()

enc = ColumnTransformerEnc(dataset.descriptor)
enc.target_encoder = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

generator = GeneticGenerator(bbox, dataset, enc)
surrogate = DecisionTreeSurrogate()

# Initialize Lore
tabularLore = Lore(bbox, dataset, enc, generator, surrogate)

instance_id = 7  # You can pick any row index
# Drop the target column 'default' before passing to explain
instance = dataset.df.drop(columns='default').iloc[instance_id]
explanation = tabularLore.explain(instance)
print(explanation)

def get_features_from_rule(explanation):
    return {premise['attr'] for premise in explanation['rule']['premises']}

# Example usage
features_used = get_features_from_rule(explanation)
print("Features used in the explanation (main rule):", features_used)

prediction = model.predict([instance])
print(prediction)
# Step 1: Get predicted probabilities
proba = model.predict_proba([instance])[0]  # Gives probabilities for each class
print(proba)
# Step 2: Get predicted class
predicted_class = model.predict([instance])
print(predicted_class)
# Step 3: Get index of predicted class
class_index = list(model.classes_).index(predicted_class)
print(class_index)

# Step 4: Get confidence score
original_confidence_score = proba[class_index]
print(original_confidence_score)
import numpy as np

# Features to perturb
features_to_perturb = features_used

# Original instance
instance1 = dataset.df.iloc[instance_id].copy()
original_instance = instance1.drop(labels='default')

# Predict original class
original_pred_class = model.predict([original_instance])[0]
print(f"Original predicted class: {original_pred_class}")

def get_random_value_excluding(column, exclude_value):
    col_data = dataset.df[column].dropna()

    # Numerical feature
    if pd.api.types.is_numeric_dtype(col_data):
        # Calculate IQR
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1

        # Define IQR range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter values in IQR range (excluding the original value)
        filtered_values = col_data[(col_data >= lower_bound) & 
                                   (col_data <= upper_bound) & 
                                   (col_data != exclude_value)]

        if filtered_values.empty:
            return exclude_value  # fallback

        return np.random.choice(filtered_values)
    
    # Categorical feature
    else:
        unique_values = col_data.astype(str).unique()
        filtered_values = [val for val in unique_values if val != str(exclude_value)]
        
        if not filtered_values:
            return exclude_value  # fallback

        return np.random.choice(filtered_values)

# Lists for tracking
confidence_scores = []
class_changes = 0
n_iterations = 100

for i in range(n_iterations):
    # Perturb the instance
    perturbed_instance = original_instance.copy()
    for feature in features_to_perturb:
        perturbed_instance[feature] = get_random_value_excluding(feature, original_instance[feature])
    
    perturbed_df = pd.DataFrame([perturbed_instance])

    # Predict class and confidence
    predicted_proba = model.predict_proba(perturbed_df)[0]
    predicted_class = model.predict(perturbed_df)[0]
    confidence = predicted_proba[predicted_class]

    confidence_scores.append(confidence)

    # Check and report class change
    if predicted_class != original_pred_class:
        class_changes += 1
        print(f"[{i+1}] Class changed to: {predicted_class} with confidence: {confidence:.4f}")

# Summary
print(f"\nOut of {n_iterations} perturbations:")
print(f"- Class changed {class_changes} times")
print(f"- Mean confidence: {np.mean(confidence_scores):.4f}")
print(f"Min: {np.min(confidence_scores):.4f}, Max: {np.max(confidence_scores):.4f}")

# Difference from original
differences = [original_confidence_score - c for c in confidence_scores]
print(f"\nAverage drop in confidence vs. original: {np.mean(differences):.4f}")

def get_features_from_rule(explanation):
    return {premise['attr'] for premise in explanation['rule']['premises']}

# Get features used in the explanation
features_used = get_features_from_rule(explanation)

# Get all feature names from the dataset (excluding target column 'default')
all_features = dataset.df.drop(columns='default').columns

# Get features not used in the explanation
unused_features = [feature for feature in all_features if feature not in features_used]

print("Features NOT used in the explanation:", unused_features)
