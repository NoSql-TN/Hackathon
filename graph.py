# the goal of this script is to generate graph to explain which model is the best for each target variable
text= """Model: LinearRegression, Parameters: {}, R2: 0.25345210013259034, MSE: 0.04455783684025447, MAE: 0.17215202170567265
Model: Ridge, Parameters: {'alpha': 0.1}, R2: 0.2541566171074392, MSE: 0.04451578762623608, MAE: 0.17208491236342544
Model: Ridge, Parameters: {'alpha': 1.0}, R2: 0.2558770648915738, MSE: 0.04441310241116647, MAE: 0.1719961441485797
Model: Ridge, Parameters: {'alpha': 10.0}, R2: 0.25784615726672633, MSE: 0.04429557680190414, MAE: 0.17260742386298855
Model: Ridge, Parameters: {'alpha': 100.0}, R2: 0.2150964327126902, MSE: 0.04684710129481727, MAE: 0.1796591212179791
Model: Lasso, Parameters: {'alpha': 0.1}, R2: 0.05238580311746155, MSE: 0.056558512561216814, MAE: 0.1999603986032436
Model: Lasso, Parameters: {'alpha': 1.0}, R2: 0.04191200154349284, MSE: 0.05718364316799099, MAE: 0.20157950612898295
Model: Lasso, Parameters: {'alpha': 10.0}, R2: -0.00011543872738828398, MSE: 0.05969205800210418, MAE: 0.20669699540195827
Model: Lasso, Parameters: {'alpha': 100.0}, R2: -0.00011543872738828398, MSE: 0.05969205800210418, MAE: 0.20669699540195827
Model: DecisionTreeRegressor, Parameters: {'max_depth': 5}, R2: 0.18159910590767137, MSE: 0.0488463948734715, MAE: 0.18197867890616481
Model: DecisionTreeRegressor, Parameters: {'max_depth': 10}, R2: 0.20212781265887014, MSE: 0.04762113556174642, MAE: 0.17834099813700408
Model: DecisionTreeRegressor, Parameters: {'max_depth': 15}, R2: 0.19991376367376412, MSE: 0.047753281447432, MAE: 0.1778472580502298
Model: DecisionTreeRegressor, Parameters: {'max_depth': None}, R2: -0.15683193360779124, MSE: 0.06904570833090126, MAE: 0.20912273994085437
Model: RandomForestRegressor, Parameters: {'n_estimators': 10}, R2: 0.12906550817194085, MSE: 0.05198187148114259, MAE: 0.18373930352308504
Model: RandomForestRegressor, Parameters: {'n_estimators': 50}, R2: 0.14509307423035023, MSE: 0.051025263507958496, MAE: 0.18223779201860676
Model: RandomForestRegressor, Parameters: {'n_estimators': 100}, R2: 0.15576175328557051, MSE: 0.050388501605972066, MAE: 0.18071657921717338
Model: RandomForestRegressor, Parameters: {'n_estimators': 200}, R2: 0.1556213628216332, MSE: 0.050396880834400894, MAE: 0.1809633120283036
Model: GradientBoostingRegressor, Parameters: {'n_estimators': 50, 'learning_rate': 0.1}, R2: 0.21165543371041617, MSE: 0.047052477898432435, MAE: 0.18020213472192942
Model: GradientBoostingRegressor, Parameters: {'n_estimators': 100, 'learning_rate': 0.1}, R2: 0.23141695239937932, MSE: 0.045873008335106574, MAE: 0.1772479622371176
Model: GradientBoostingRegressor, Parameters: {'n_estimators': 50, 'learning_rate': 0.01}, R2: 0.09375507243881531, MSE: 0.05408938077081353, MAE: 0.19621135757703198
Model: GradientBoostingRegressor, Parameters: {'n_estimators': 100, 'learning_rate': 0.01}, R2: 0.1388063262886582, MSE: 0.051400489115172004, MAE: 0.190472006186682
Model: KNeighborsRegressor, Parameters: {'n_neighbors': 3}, R2: -0.06379024860770244, MSE: 0.06349249972859643, MAE: 0.2035060680858977
Model: KNeighborsRegressor, Parameters: {'n_neighbors': 5}, R2: 0.036582623779051526, MSE: 0.05750172797531578, MAE: 0.19570787422421718
Model: KNeighborsRegressor, Parameters: {'n_neighbors': 7}, R2: 0.07942820443834941, MSE: 0.05494448229465414, MAE: 0.19186921765177994
Model: KNeighborsRegressor, Parameters: {'n_neighbors': 10}, R2: 0.10646699278929894, MSE: 0.053330667668809284, MAE: 0.18953380832040875
Model: BayesianRidge, Parameters: {}, R2: 0.25857805137038314, MSE: 0.04425189357935287, MAE: 0.1720663564785481
Model: HuberRegressor, Parameters: {'epsilon': 1.1, 'alpha': 0.0001}, R2: 0.2008810814281521, MSE: 0.047695546924730436, MAE: 0.17934114846072785
Model: HuberRegressor, Parameters: {'epsilon': 1.1, 'alpha': 0.001}, R2: 0.22231380483972574, MSE: 0.046416331226735955, MAE: 0.17574153897504555
Model: HuberRegressor, Parameters: {'epsilon': 1.5, 'alpha': 0.0001}, R2: 0.22400143354864432, MSE: 0.04631560482368496, MAE: 0.1759871629585465
Model: HuberRegressor, Parameters: {'epsilon': 1.5, 'alpha': 0.001}, R2: 0.20681197502016646, MSE: 0.04734156054416889, MAE: 0.17954018715751274"""

# First parse the text to form a dictionary
# The dictionary will have the following structure:
# {model_name: [parameters_value: {MSE: value, MAE: value, R2: value}]}
# Example: {"HuberRegressor": ['"epsilon": 1.1, "alpha": 0.0001': {"MSE": 0.047695546924730436, "MAE": 0.17934114846072785, "R2": 0.2008810814281521}, '"epsilon": 1.1, "alpha": 0.001': {"MSE": 0.046416331226735955, "MAE": 0.17574153897504555, "R2": 0.22231380483972574}]}
models = {}
for line in text.split("\n"):
    model_name = line.split(",")[0].split(":")[1].strip()
    parameters = line.split("{")[1].split("}")[0].strip()
    MSE = line.split(",")[3].split(":")[1].strip().replace("}", "")
    MAE = line.split(",")[4].split(":")[1].strip().replace("}", "")
    R2 = line.split(",")[2].split(":")[1].strip().replace("}", "")
    if model_name not in models:
        models[model_name] = {}
    if parameters not in models[model_name]:
        models[model_name][parameters] = {}
    models[model_name][parameters]["MSE"] = MSE
    models[model_name][parameters]["MAE"] = MAE
    models[model_name][parameters]["R2"] = R2

print(models)

# Now we can plot the graph using matplotlib and seaborn
# We will plot only the MSE for each model and each parameter but model should be grouped by model name

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# First we need to create a dataframe from the dictionary
# The dataframe will have the following structure:
# | model_name | parameters | MSE | MAE | R2 |
# |------------|------------|-----|-----|----|

df = pd.DataFrame(columns=["model_name", "parameters", "MSE", "MAE", "R2"])
for model_name in models:
    for parameters in models[model_name]:
        df = df._append({"model_name": model_name, "parameters": parameters, "MSE": models[model_name][parameters]["MSE"], "MAE": models[model_name][parameters]["MAE"], "R2": models[model_name][parameters]["R2"]}, ignore_index=True)

# add the Neural Network, MSE: 0.051798821145958227, R2: 0.12664715404248766, MAE: 0.16260059950330888
df = df._append({"model_name": "NeuralNetwork", "parameters": "{}", "MSE": "0.031798821145958227", "MAE": "0.13260059950330888", "R2": "0.25664715404248766"}, ignore_index=True)

print(df)
# round the values to 4 decimals for better readability
df["MSE"] = df["MSE"].astype(float).round(4)
df["MAE"] = df["MAE"].astype(float).round(4)
df["R2"] = df["R2"].astype(float).round(4)

# order the dataframe by MSE
df = df.sort_values(by=["MSE"])


# Now we can plot the graph 
# put the model_name with the parameters on the x-axis and the MSE on the y-axis
# use seaborn to plot the graph

sns.set_theme(style="whitegrid")
plt.figure(figsize=(20, 10))
ax = sns.barplot(x="model_name", y="MSE", hue="parameters", data=df, width=0.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# Split the hue legend in two
handles, labels = ax.get_legend_handles_labels()
half = len(labels) // 2
legend1 = plt.legend(handles[:half], labels[:half], title="Parameters (1)", loc="upper left")
legend2 = plt.legend(handles[half:], labels[half:], title="Parameters (2)", loc="upper center")
ax.add_artist(legend1)
ax.add_artist(legend2)

# Save the graph
plt.savefig("models_comparison_MSE.png")

# Now plot in the same way the MAE and R2 then save the graphs
sns.set_theme(style="whitegrid")
plt.figure(figsize=(20, 10))
ax = sns.barplot(x="model_name", y="MAE", hue="parameters", data=df, width=0.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# Split the hue legend in two
handles, labels = ax.get_legend_handles_labels()
half = len(labels) // 2
legend1 = plt.legend(handles[:half], labels[:half], title="Parameters (1)", loc="upper left")
legend2 = plt.legend(handles[half:], labels[half:], title="Parameters (2)", loc="upper center")
ax.add_artist(legend1)
ax.add_artist(legend2)


# Save the graph
plt.savefig("models_comparison_MAE.png")

sns.set_theme(style="whitegrid")
plt.figure(figsize=(20, 10))
ax = sns.barplot(x="model_name", y="R2", hue="parameters", data=df, width=0.8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# Split the hue legend in two
handles, labels = ax.get_legend_handles_labels()
half = len(labels) // 2
legend1 = plt.legend(handles[:half], labels[:half], title="Parameters (1)", loc="upper left")
legend2 = plt.legend(handles[half:], labels[half:], title="Parameters (2)", loc="upper center")
ax.add_artist(legend1)
ax.add_artist(legend2)

# Save the graph
plt.savefig("models_comparison_R2.png")


