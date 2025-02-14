import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Generating synthetic company-related dataset
np.random.seed(42)
num_records = 1200  # Number of observations

enhanced_company_data = {
    "Month": pd.date_range(start="2018-01-01", periods=num_records, freq='M').strftime('%Y-%m'),
    "Production_Output": np.random.randint(5000, 20000, num_records),
    "Defective_Rate": np.random.uniform(0.5, 5.0, num_records),
    "Supply_Chain_Delays": np.random.randint(0, 10, num_records),
    "Raw_Material_Cost": np.random.randint(50000, 150000, num_records),
    "Energy_Consumption": np.random.randint(10000, 50000, num_records),
    "AI_Optimized_Production": np.random.choice([0, 1], num_records, p=[0.7, 0.3]),
    "Marketing_Spend": np.random.randint(50000, 300000, num_records),
    "Employee_Count": np.random.randint(1000, 5000, num_records),
    "Revenue": np.random.randint(500000, 2000000, num_records)
}

df = pd.DataFrame(enhanced_company_data)

# Introducing some missing values
for col in ["Production_Output", "Defective_Rate", "Energy_Consumption", "Marketing_Spend"]:
    df.loc[np.random.choice(df.index, size=100, replace=False), col] = np.nan

# Introducing inconsistent values
df.loc[np.random.choice(df.index, size=20, replace=False), "AI_Optimized_Production"] = 2  # Should be 0 or 1

# Data Wrangling
# Fixing inconsistent AI_Optimized_Production values
df["AI_Optimized_Production"] = df["AI_Optimized_Production"].replace(2, 1)

# Handling missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
df[["Production_Output", "Defective_Rate", "Energy_Consumption", "Marketing_Spend"]] = imputer.fit_transform(
    df[["Production_Output", "Defective_Rate", "Energy_Consumption", "Marketing_Spend"]]
)

# Converting categorical time feature to numerical format
df["Month_Num"] = np.arange(1, len(df) + 1)

# Splitting the dataset into features and target variable
X = df.drop(columns=["Month", "Revenue"])
y = df["Revenue"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building and Training the Random Forest Regressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Making Predictions
y_pred = model.predict(X_test_scaled)

# Evaluating the model
r2_score_result = r2_score(y_test, y_pred)
print(f"R² Score: {r2_score_result}")
