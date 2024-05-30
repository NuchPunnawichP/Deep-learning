import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# I am using my big brother's computer
data = r"C:\Users\Pannawit\Documents\GitHub\Deep-learning\Basic_AI_FirstLearning\Test1\Book1.xlsx"
df = pd.read_excel(data, engine='openpyxl')

features = df['x'].values.reshape(-1, 1)
labels = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

new_data = [[410], [10.2], [7.8]]
new_predictions = model.predict(new_data)

print("Predictions for new data:")
for i, prediction in enumerate(new_predictions):
    print(f"Input: {new_data[i][0]}, Prediction: {prediction}")
