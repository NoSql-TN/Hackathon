import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix

# Load your dataset
# Assuming your dataset is in a CSV file named 'your_dataset.csv'
data = pd.read_csv('Desktop/Hackaton/dataset.csv')
data = data.drop('Unnamed: 0', axis=1)
# Exclude non-numeric columns from the correlation analysis
numeric_df = data.select_dtypes(include='number')

#select only the 5000 songs with most popularity
data = data.sort_values(by=['popularity'], ascending=False)
data = data.head(5000)


# Display the correlation matrix
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Display the correlation with the target variable (popularity)
popularity_correlation = correlation_matrix['popularity'].sort_values(ascending=False)
print("Correlation with Popularity:\n", popularity_correlation)

# Select the most highly correlated features
# (you may alternatively select the features manually)
threshold = 0.5
popularity_correlation = popularity_correlation[popularity_correlation > threshold]
print("Most correlated features:\n", popularity_correlation)

# The problem that appears here is that the correlation between the features and the popularity is not very high.
# The Maximum is 0.05 and the lowest is -0.10 so we will probably need to find an other way to predict the popularity


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select features (excluding non-numeric columns like track_id, artists, etc.)
X = data.select_dtypes(include='number').drop(columns=['popularity'])

# Target variable
y = data['popularity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)


# Make a code able to predict the popularity of a song if we give the features of the song

print("Popularity prediction for a song:")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming your dataframe is named 'df'
# Perform data preprocessing, feature selection, and conversion to numerical values

# Select features and target variable
X = data.drop(['popularity', 'track_id'], axis=1)
y = data['popularity']

# Separate numerical and categorical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()







#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rf2=RandomForestClassifier(n_estimators=100, max_depth = 100, max_features = 'sqrt', min_samples_leaf =1,min_samples_split =2)

#Train the model using the training sets y_pred=clf.predict(X_test)
rf2.fit(X_train, y_train)

y_train_predrf2 = rf2.predict(X_train)
y_test_predrf2 = rf2.predict(X_test)

conf_matrix_rf2 = confusion_matrix(y_true=y_test, y_pred=y_test_predrf2)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix_rf2, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix_rf2.shape[0]):
    for j in range(conf_matrix_rf2.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix_rf2[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print("Accuracy: ", (conf_matrix_rf2[0,0]+conf_matrix_rf2[1,1])/(conf_matrix_rf2[0,0]+conf_matrix_rf2[1,1]+conf_matrix_rf2[0,1]+conf_matrix_rf2[1,0]) )
print("Precision: ", (conf_matrix_rf2[0,0])/(conf_matrix_rf2[0,0]+conf_matrix_rf2[0,1]) )
print("Recall: ", (conf_matrix_rf2[0,0])/(conf_matrix_rf2[0,0]+conf_matrix_rf2[1,0]) )


