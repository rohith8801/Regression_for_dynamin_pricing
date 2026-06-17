   #IMPORTING DATASET
import pandas as pd
import numpy as np
import pickle

url = "https://huggingface.co/datasets/gradio/NYC-Airbnb-Open-Data/resolve/main/AB_NYC_2019.csv"
df = pd.read_csv(url)
# print(df.head(5))


    # CLEANING AND DATASET PRE-PROCESSING
# ---------- REALISTIC FEATURE CREATION ----------

np.random.seed(42)
# 1. AC (0 = No, 1 = Yes)
df["AC"] = np.random.randint(0, 2, len(df))
# 2. Furnishing level (1 = low, 2 = medium, 3 = high)
df["furnishing"] = np.random.randint(1, 4, len(df))
# 3. Luxury score (based on location + randomness)
df["luxury_score"] = (
    (df["neighbourhood_group"] == "Manhattan").astype(int) * 2 +
    np.random.rand(len(df))
)
# 4. Room size (approx using reviews + noise)
df["room_size"] = df["number_of_reviews"] * 0.1 + np.random.rand(len(df)) * 10
# 5. Amenities score (0–10 scale)
df["amenities_score"] = (
    df["availability_365"] / 36 +
    df["reviews_per_month"].fillna(0) * 2 +
    np.random.rand(len(df)) * 2
)
# 6. Demand score
df["demand_score"] = (
    df["number_of_reviews"] * 0.5 +
    df["reviews_per_month"].fillna(0) * 10
)

df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

cols = [
    "price",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "reviews_per_month",
    "latitude",
    "longitude",
    "room_type",
    "neighbourhood_group",
    "AC",
    "furnishing",
    "luxury_score",
    "room_size",
    "amenities_score",
    "demand_score"
]
df = df[cols].copy()
df = pd.get_dummies(df, columns=["room_type", "neighbourhood_group"], drop_first=True)  #ENCODING VALUES

# ---------- PRICE FORMULA ----------

# Base price
df["base_price"] = 50 + df["room_size"] * 2 + df["furnishing"] * 20
# Demand
df["demand_factor"] = 1 + (df["demand_score"] / df["demand_score"].max())
# Supply
df["supply_factor"] = 1 - (df["availability_365"] / 365) * 0.3
# Quality
df["quality_factor"] = (
    1 +
    df["amenities_score"] * 0.05 +
    df["AC"] * 0.1 +
    df["furnishing"] * 0.1
)
# Location
df["location_factor"] = (
    1 +
    df.get("neighbourhood_group_Manhattan", 0) * 0.5 +
    df.get("neighbourhood_group_Brooklyn", 0) * 0.2
)
# Final price
df["price"] = (
    df["base_price"] *
    df["demand_factor"] *
    df["supply_factor"] *
    df["quality_factor"] *
    df["location_factor"]
)
# Add noise
import numpy as np
df["price"] = df["price"] + np.random.normal(0, 20, len(df))

X = df.drop("price", axis=1)
y = df["price"]



    # TRAIN_TEST_SPLIT
from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train ,y_test = train_test_split(X ,y, test_size=0.2,random_state=42)



    # MODEL IMPLEMENTATION
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=3,
    learning_rate=0.05
)
model.fit(X_train , y_train)



    # PREDICTIONS OF MODEL
y_train_pred = model.predict(X_train)
print("train predictions :", y_train_pred)
y_test_pred = model.predict(X_test)
print("train predictions :", y_test_pred)


    # MODEL EVALUATION
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

train_mse = mean_squared_error(y_train , y_train_pred)
print("train mse:",train_mse)
test_mse = mean_squared_error(y_test , y_test_pred)
print("test mse:",test_mse)

train_rmse = np.sqrt(mean_squared_error(y_train , y_train_pred))
print("train rmse:",train_rmse)
test_rmse = np.sqrt(mean_squared_error(y_test , y_test_pred))
print("test rmse:",test_rmse)

train_r2 = r2_score(y_train , y_train_pred)
print("train r2:",train_r2)
test_r2 = r2_score(y_test , y_test_pred)
print("test r2:",test_r2)


# SAVING THE MODEL
import os

os.makedirs("model", exist_ok=True)
import pickle

pickle.dump(X.columns, open("model/columns.pkl", "wb"))
pickle.dump(model, open("model/price_model.pkl", "wb"))

print("model saved")