# Importing required libraries
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Define product categories, brands, and stock availability
categories = ['Men', 'Women', 'Kids', 'Accessories', 'Footwear', 'Home']
brands = ['Nike', 'Adidas', 'Puma', 'Levis', 'Pepe Jeans', 'Biba', 'Van Heusen', 'H&M']
stock_availability = ['In stock', 'Out of stock']

# Generate synthetic data
np.random.seed(42)  # For reproducibility
data = {
    'Product_ID': [f'P{str(i).zfill(4)}' for i in range(1, 1001)],  # Generating simple product IDs like P0001, P0002, etc.
    'Category': [random.choice(categories) for _ in range(1000)],
    'Brand': [random.choice(brands) for _ in range(1000)],
    'Price': [random.uniform(500, 5000) for _ in range(1000)],
    'Ratings': [random.uniform(3.0, 5.0) for _ in range(1000)],
    'Reviews': [random.randint(50, 10000) for _ in range(1000)],
    'Discount': [random.randint(5, 50) for _ in range(1000)],
    'Stock_Availability': [random.choice(stock_availability) for _ in range(1000)],
}

# Create DataFrame
df = pd.DataFrame(data)

# Show the first few rows of the dataset
print(df.head())

# ----------------------------------------
# 1. Exploratory Data Analysis (EDA)
# ----------------------------------------

# 1.1 Box Plot: Price Distribution by Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Price', data=df)
plt.title('Price Distribution by Category')
plt.xticks(rotation=45)
plt.show()

# 1.2 Histogram: Distribution of Product Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30, kde=True, color='skyblue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 1.3 Scatter Plot: Price vs Ratings
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Price', y='Ratings', data=df, color='orange')
plt.title('Price vs Ratings')
plt.xlabel('Price')
plt.ylabel('Ratings')
plt.show()

# 1.4 Pearson Correlation Matrix
correlation_matrix = df[['Price', 'Ratings', 'Reviews', 'Discount']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Pearson Correlation Matrix')
plt.show()

# Show correlation values
print(correlation_matrix)

# ----------------------------------------
# 2. Data Preprocessing
# ----------------------------------------

# Convert categorical columns to dummy variables (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['Category', 'Brand', 'Stock_Availability'], drop_first=True)

# Define features (X) and target variable (y)
X = df_encoded.drop(['Price', 'Product_ID'], axis=1)  # Independent variables
y = df_encoded['Price']  # Dependent variable (target)

# Display the processed data
print(X.head())

# ----------------------------------------
# 3. Model Training and Evaluation
# ----------------------------------------

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# ----------------------------------------
# 4. Model Visualization
# ----------------------------------------

# Scatter plot of actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()