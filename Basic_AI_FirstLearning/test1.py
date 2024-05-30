import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = r"C:\Users\Pannawit\Documents\GitHub\Deep-learning\Basic_AI_FirstLearning\Book1.xlsx"

df = pd.read_excel(data, engine='openpyxl')

features = df[['x']].values
labels = df['y'].values

regressor = DecisionTreeRegressor()
regressor = regressor.fit(features, labels)

test = 45
prediction = regressor.predict([[test]])

print("Prediction:", prediction)
