import pandas as pd
from Generate_encoded_lyrics import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import pickle

# Load the data
csv = pd.read_csv('csv/spotify_song.csv')
print("X Loaded")

csv = csv[csv['parole'].notna()]

X = csv[['parole']]
Y= csv[['popularity']]

print(X)
print(Y)
# Preprocess the data
X = X['parole'].apply(preprocess_text)
X = X.astype(str)
lyrics_encoder = pickle.load(open('models/lyrics_pop_encoder.pkl', 'rb'))
lyrics_encoded = lyrics_encoder.transform(X).toarray()
lyrics_encoded = pd.DataFrame(lyrics_encoded, columns=lyrics_encoder.get_feature_names_out())
X = lyrics_encoded
print(X)
print("Lyrics encoded")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=27)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.dtypes, X_test.dtypes, y_train.dtypes, y_test.dtypes)


# def kerasAlgorithm():
#     # Algorithms to test
#     algorithms = [
#         LinearRegression,
#         Ridge,
#         Lasso,
#         DecisionTreeRegressor,
#         RandomForestRegressor,
#         GradientBoostingRegressor,
#         KNeighborsRegressor,
#         BayesianRidge,
#         HuberRegressor
#     ]

#     # Parameters to test for each model
#     params = {
#         'LinearRegression': [{}],
#         'Ridge': [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}, {'alpha': 100.0}],
#         'Lasso': [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}, {'alpha': 100.0}],
#         'SVR': [{'kernel': 'linear', 'C': 0.1}, {'kernel': 'linear', 'C': 1.0}, 
#                 {'kernel': 'rbf', 'C': 0.1, 'gamma': 'auto'}, {'kernel': 'rbf', 'C': 1.0, 'gamma': 'auto'}],
#         'DecisionTreeRegressor': [{'max_depth': 5}, {'max_depth': 10}, {'max_depth': 15}, {'max_depth': None}],
#         'RandomForestRegressor': [{'n_estimators': 10}, {'n_estimators': 50}, {'n_estimators': 100}, {'n_estimators': 200}],
#         'GradientBoostingRegressor': [{'n_estimators': 50, 'learning_rate': 0.1},
#                                     {'n_estimators': 100, 'learning_rate': 0.1},
#                                     {'n_estimators': 50, 'learning_rate': 0.01},
#                                     {'n_estimators': 100, 'learning_rate': 0.01}],
#         'KNeighborsRegressor': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}, {'n_neighbors': 10}],
#         'BayesianRidge': [{}],
#         'HuberRegressor': [{'epsilon': 1.1, 'alpha': 0.0001}, {'epsilon': 1.1, 'alpha': 0.001},
#                         {'epsilon': 1.5, 'alpha': 0.0001}, {'epsilon': 1.5, 'alpha': 0.001}]
#     }

#     best_models = {}

#     for target_variable in tqdm(Y.columns, desc="Target Variable Progress"):
#         best_mse = 99999
#         best_model = None
#         best_params = None

#         for model_class in tqdm(algorithms, desc="Models Progress", leave=False):
#             model_name = model_class().__class__.__name__

#             for param_set in tqdm(params[model_name], desc=f"Parameters for {model_name}", leave=False):
#                 # Create model instance with specified parameters
#                 model = model_class(**param_set)

#                 # Train the model
#                 model.fit(X_train, y_train[target_variable])

#                 # Predictions
#                 y_pred = model.predict(X_test)

#                 # Evaluate the model
#                 r2 = r2_score(y_test[target_variable], y_pred)
#                 mse = mean_squared_error(y_test[target_variable], y_pred)
#                 mae = mean_absolute_error(y_test[target_variable], y_pred)
                
#                 print(f"Model: {model_name}, Parameters: {param_set}, R2: {r2}, MSE: {mse}, MAE: {mae}")

#                 # Check if current model is better than the best one
#                 if mse < best_mse:
#                     best_mse = mse
#                     best_model = model
#                     best_params = param_set

#         best_models[target_variable] = {'model': best_model, 'params': best_params, 'mse': best_mse}

#     # Save the best models for each variable
#     for target_variable, model_info in best_models.items():
#         model = model_info['model']
#         params = model_info['params']
#         mse = model_info['mse']

#         print(f"Best Model for {target_variable}: {model.__class__.__name__} with Parameters {params} and Mse: {mse}")

#         # Save the model if it is good enough
#         if r2 > 0.5:
#             pickle.dump(model, open(f'best_model_{target_variable}_{model.__class__.__name__}.pkl', 'wb'))
#             print(f"Best Model for {target_variable} saved")


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm
import random


# Convert Pandas DataFrames to NumPy arrays
X_train_nn, X_test_nn = X_train.values, X_test.values
y_train_nn, y_test_nn = y_train.values, y_test.values


# Define the neural network model
def create_popularity_predictor(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  

    # Compile the model
    optimizer = Adam(learning_rate=0.001)  
    model.compile(loss='mae', optimizer=optimizer, metrics=['mse'])

    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

# Train and evaluate the neural network
input_dim = X_train_nn.shape[1]
model_popularity = create_popularity_predictor(input_dim)
history = model_popularity.fit(X_train_nn, y_train_nn, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)

# Predictions
y_pred_nn = model_popularity.predict(X_test_nn)

# Evaluate the model
mse_nn = mean_squared_error(y_test_nn, y_pred_nn)
r2_nn = r2_score(y_test_nn, y_pred_nn)
mae_nn = mean_absolute_error(y_test_nn, y_pred_nn)

print(f"Neural Network, MSE: {mse_nn}, R2: {r2_nn}, MAE: {mae_nn}")

# Show 5 random predictions and the real values
for i in range(5):
    idx = random.randint(0, len(y_test_nn) - 1)
    print(f"Predicted: {y_pred_nn[i]}, Real: {y_test_nn[i]}")


# Save the model
model_popularity.save('best_model_popularity_nn.keras')
print("Best Neural Network Model saved")

