import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Inspect the dataset
print(data.head())
print(data.info())

# Data preprocessing
# Handle missing values
data = data.dropna()

# Feature engineering (if needed, depends on the dataset)
# For example, extracting month and year from a date column, if present
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data = data.drop(columns=['date'])

# Split the data into features and target variable
X = data.drop(columns=['sales'])  # Assuming 'sales' is the target variable
y = data['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')

# Forecasting
forecast = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Forecast': forecast})

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(results['Actual'].values, label='Actual')
plt.plot(results['Forecast'].values, label='Forecast')
plt.legend()
plt.show()
