
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
import xgboost as xgb
import shap
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.colors as mcolors

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Load your dataset
data = pd.read_csv('/Users/nell/Desktop/paper 2024/Dataset to use.csv', low_memory=False)

# Fill null values with 0
data.fillna(0, inplace=True)

# Convert new categorical variables to binary format
categorical_cols = ['Vegetation', 'Presence of Fences',
                    'Signboards (%)']
for col in categorical_cols:
    data[col] = data[col].apply(lambda x: 1 if x > 0 else 0)

# Features and target
features = ['Visible Sky (%)', 'Buildings (%)', 'Visible Road (%)',
             'Signboards (%)', 'Vegetation', 'Presence of Fences',
            'Presence of Sidewalks',
            'Road Types', 'Post Speed', 'Traffic Direction', 'Number of Lanes', 'Traffic Volume (Thousands per day)',
           'Land Use ', 'Asian Population (%)', 'Black Population (%)','White Population (%)','Hispanic Population (%)',
'Household Income 150000 to 199999 (%)',
'Household Income 200000 and more (%)',
'Household Income less than 10000 (%)']
target = 'y'

X = data[features]
y = data[target]

# Create dummy variables for the categorical columns including the new ones
X = pd.get_dummies(X, columns=['Road Types', 'Presence of Sidewalks','Post Speed', 'Traffic Direction', 'Number of Lanes','Land Use '], drop_first=False)
# Balance the dataset
group_0_indices = y[y == 0].sample(n=9000, random_state=random_seed).index
group_1_indices = y[y == 1].sample(n=9000, random_state=random_seed).index

X_group_0 = X.loc[group_0_indices]
y_group_0 = y.loc[group_0_indices]

X_group_1 = X.loc[group_1_indices]
y_group_1 = y.loc[group_1_indices]

X_sampled = pd.concat([X_group_0, X_group_1])
y_sampled = pd.concat([y_group_0, y_group_1])

# Split the balanced data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=random_seed)

# Hyperparameter tuning
hyperparameter_grid = {
    'max_depth': [3, 4, 5, 8],
    'learning_rate': [0.1, 0.01, 0.001]
}

best_score = float('-inf')
best_params = None

for max_depth in hyperparameter_grid['max_depth']:
    for learning_rate in hyperparameter_grid['learning_rate']:
        model = xgb.XGBClassifier(objective="binary:logistic", max_depth=max_depth, learning_rate=learning_rate)
        scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
        accuracy = np.mean(scores)

        if accuracy > best_score:
            best_score = accuracy
            best_params = {'max_depth': max_depth, 'learning_rate': learning_rate}

print("Best Hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = xgb.XGBClassifier(objective="binary:logistic", max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'])
final_model.fit(X_train, y_train)

# Predict on the test set with the final model
y_pred = final_model.predict(X_test)

explainer = shap.Explainer(final_model)
shap_values = explainer(X_test)
# Print all column names after converting to dummy variables to check their spelling
print("Column names after converting to dummy variables:")
print(X.columns)

# Check unique values in the 'Land Use ' column
print(data['Land Use '].unique())
# Calculate the mean absolute SHAP values for each feature
mean_abs_shap_values = np.abs(shap_values.values).mean(0)
sorted_feature_indices_by_mean = np.argsort(mean_abs_shap_values)[::-1]  # Reverse to get descending order

# List of specific features you want to plot
selected_features = []

# Get the indices of the selected features
specific_feature_indices = [X_test.columns.get_loc(feature) for feature in selected_features]

# Combine specific features with other top features to make up to 20
all_feature_indices = list(sorted_feature_indices_by_mean)
# Ensure specific features are included at the top of the list
all_feature_indices = specific_feature_indices + [idx for idx in all_feature_indices if idx not in specific_feature_indices]
# Limit to top 20 features
top_feature_indices = all_feature_indices[:20]

# Get the feature names for the top 20 features
top_feature_names = X_test.columns[top_feature_indices]

# Generate the beeswarm plot for the top 20 features
shap.summary_plot(shap_values.values[:, top_feature_indices], features=X_test.iloc[:, top_feature_indices].values, feature_names=top_feature_names)

# Display the plot
#plt.show()

# Scatter Plot
# List of features to plot
features_to_plot = ['Household Income 200000 and more (%)','Land Use _Residential']

# Generate SHAP Dependence plots for each of the features in the list
for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[feature], shap_values.values[:, X_test.columns.get_loc(feature)], c='brown', alpha=0.6)
    plt.title(f"SHAP Dependence Plot: {feature}")
    plt.xlabel(feature)
    plt.ylabel(f"SHAP value for {feature}")
  #  plt.show()

# Custom colormap from brown to orange
colors = ["orange", "brown"]
cmap_brown_orange = mcolors.LinearSegmentedColormap.from_list("brown_orange", colors)

# Adjust font size and line width globally
plt.rcParams.update({'font.size': plt.rcParams['font.size'] + 2, 'lines.linewidth': plt.rcParams['lines.linewidth'] + 0.5})

# SHAP Dependence plot of 'Household Income 200000 and more (%)' with respect to 'Land Use _Commercial'
shap.dependence_plot('Household Income 200000 and more (%)', shap_values.values, X_test,
                     interaction_index='Land Use _Commercial', cmap=cmap_brown_orange, show=False)

# Set the title with an increased font size
#plt.title('SHAP Dependence of "Income" with Respect to "Land Use"', fontsize=plt.rcParams['font.size'])

# Show the plot
#plt.show()
# Evaluate the final model for each class
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy:.2f}\n")

# Get precision, recall, f1-score for each class without averaging
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)

# Get the unique list of classes
classes = np.unique(y_test)

# Print precision, recall, f1-score for each class
print("Classification Metrics for each class:")
for i, class_label in enumerate(classes):
    print(f"Class {class_label}:")
    print(f"Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1-Score: {f1_score[i]:.2f}, Support: {support[i]}")
    print()

# Correctly predict using final_model and calculate the confusion matrix
y_pred = final_model.predict(X_test)  # Corrected to use final_model for prediction

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate percentages for the confusion matrix to understand the distribution better
cm_percentage = cm / np.sum(cm) * 100

# Labels for each quadrant of the confusion matrix
labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
labels = np.asarray(labels).reshape(2,2)

# Create a heatmap for the confusion matrix without percentages in the automatic annotation
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=False, fmt=".2f", ax=ax, cmap='Blues', cbar_kws={'label': 'Count'})

# Manually annotate each cell with custom text
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]  # Direct count of occurrences
        percentage = cm_percentage[i, j]  # Percentage for better understanding
        label = labels[i, j]  # Label for each quadrant
        # Annotation text includes label, count, and percentage
        text = f"{label}\n{count}\n({percentage:.2f}%)"
        plt.text(j+0.5, i+0.5, text, ha="center", va="center", color="black")

# Set axis labels and titles
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['0', '1'])
ax.yaxis.set_ticklabels(['0', '1'])

plt.show()




