import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pickle


# Load your dataset
db = pd.read_csv('dataset.csv', sep=',')
df = pd.DataFrame(db)
df = df.dropna()
df = df.drop_duplicates()
df = df.sort_values(by=['popularity'], ascending=False)
df = df.head(60000)

# Exclude non-numeric columns from the correlation analysis
numeric_df = df.select_dtypes(include='number')

#select only the 5000 songs with most popularity
df = df.sort_values(by=['popularity'], ascending=False)

#Function that plot the correlation matrix
def plot_correlation_matrix(df):
    corr = numeric_df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

#plot_correlation_matrix(df)
#change the track_id into 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15...
df = df.reset_index(drop=True)
print(df.head())




#Unfortunately we expected some parameters with a high correlation with the popularity but we didn't find any. (Max = 0.13 danceability)

#Maybe we can find some correlation between the parameters and the popularity if we use the clustering method

# Select relevant columns
selected_columns = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'tempo',
                    'duration_ms', 'popularity', 'track_genre', 'artists', 'explicit', 'key']

df_selected = df[selected_columns]
df_selected = df_selected.astype({'danceability': float , 'energy': float, 'acousticness': float, 'instrumentalness': float, 'tempo': float,
                    'duration_ms': int, 'popularity': int, 'track_genre': str, 'artists': str, 'explicit': bool, 'key': int})





# encode the genre column
genre_encoder = OneHotEncoder(handle_unknown='ignore')
genre_encoder.fit(df_selected[['track_genre']])
# save with pickle
pickle.dump(genre_encoder, open('genre_encoder_pop.pkl', 'wb'))
genre_encoded = genre_encoder.transform(df_selected[['track_genre']]).toarray()
genre_encoded = pd.DataFrame(genre_encoded, columns=genre_encoder.categories_)
# print the encoded genre column
print("The encoded genre column is:")
print(genre_encoded)
# Drop original categorical columns
#df_selected = df_selected.drop(['track_genre'], axis=1)
print("Vérification encoder genre")


#encode the artists column
artists_encoder = OneHotEncoder(handle_unknown='ignore')
artists_encoder.fit(df_selected[['artists']])
# save with pickle
pickle.dump(artists_encoder, open('artists_encoder_pop.pkl', 'wb'))
artists_encoded = artists_encoder.transform(df_selected[['artists']]).toarray()
artists_encoded = pd.DataFrame(artists_encoded, columns=artists_encoder.categories_)
# print the encoded artists column
print("The encoded artists column is:")
print(artists_encoded)
# Drop original categorical columns
#df_selected = df_selected.drop(['artists'], axis=1)
print("Vérification encoder artists")



#for the encoding of the explicit column we need to convert the boolean to int so we don't need a OneHotEncoder
#encode the explicit column with true becomes 1 and false becomes 0
df_selected['explicit'] = df_selected['explicit'].astype(int)
print("Vérification encoder explicit")
print(df_selected['explicit'])

print(df_selected)

print(df_selected['popularity'])

#Now all our parameters are well encoded, we can concatenate them to the dataframe
df_selected = pd.concat([df_selected, genre_encoded, artists_encoded], axis=1)
print(df_selected['popularity'])
# Drop the original categorical columns from the encoded DataFrame
df_selected = df_selected.drop(['track_genre', 'artists'], axis=1)
print("Vérification concaténation")
#some values are NaN so we need to replace them with 0
print(df_selected['popularity'])

# print the df_selected columns in order
print("V-------------------------------------------------")
print(df_selected.columns)

# look if 2Pac;Rappin' 4-Tay column is in the dataframe
print("V-------------------------------------------------")
print(df_selected["2Pac;Rappin\' 4-Tay"])

df_selected = df_selected.fillna(0)




#Train a model to predict the popularity
#Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = df_selected.drop(['popularity'], axis=1)
y = df_selected['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Vérification split")
print(X_train.head())
print(y_train.head())

#we need to convert the name of a column to string because we have a tuple in the name of a column
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)


# Write every column name in a file
with open('encoded_columns.csv', 'w') as f:
    for col in X_test.columns:
        f.write(col + '\n')

#Feature names are only supported if all input features have string names, but your input has ['str', 'tuple'] as feature name / column name types. If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.
#https://stackoverflow.com/questions/49545947/feature-names-are-only-supported-if-all-input-features-have-string-names-but-yo


#create a model and train it in order to predict the popularity
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Vérification modèle")
print(model)

#Predict the popularity
y_pred = model.predict(X_test)
print("Vérification prédiction")
print(y_pred)
print(y_pred[0])
print(y_test.iloc[0])
print(y_pred[10])
print(y_test.iloc[10])
print(y_pred[20])
print(y_test.iloc[20])

#Evaluate the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("Vérification évaluation")
print("MSE : ", mean_squared_error(y_test, y_pred))
print("MAE : ", mean_absolute_error(y_test, y_pred))
print("R2 : ", r2_score(y_test, y_pred))


#Overall, based on these metrics, thte Random Forest model seems to be performing very well on the test set. 
#The low values of MSE and MAE indicate that the predictions are generally close to the actual values, 
#and the high R-squared value suggests that the model is capturing the patterns in the data effectively.

#Can you do a cross validation to see if the model is overfitting or not?
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Vérification cross validation")
print(scores)
print("Mean cross validation score : ", scores.mean())


#All individual cross-validation scores are very close to each other, suggesting consistent performance across different subsets of the training data.
#The high mean cross-validation score (close to 1) indicates that your model generalizes well to new, unseen data. It consistently performs at a high level across different training subsets.

#save the model
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("Vérification sauvegarde modèle")

#load the model
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print("Vérification chargement modèle")
print(result)


