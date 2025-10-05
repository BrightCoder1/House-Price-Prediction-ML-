from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

df = pd.read_csv(
    "C:/Users/vs035/OneDrive/Desktop/House Prediction ML/Housing.csv")


# Encode categorical columns
le = LabelEncoder()

# df["guestroom"] = le.fit_transfrom(df["guestroom"])

for col in ["guestroom", "mainroad", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
    df[col] = le.fit_transform(df[col])

# One-hot encode bedrooms and parking
bedrooms_dummies = pd.get_dummies(df["bedrooms"], prefix="bedroom")
df = pd.concat([df, bedrooms_dummies], axis=1)

parking_dummies = pd.get_dummies(df["parking"], prefix="parking")
df = pd.concat([df, parking_dummies], axis=1)

# Prepare features and target
X = df.drop(["price"], axis=1)
y = df["price"]

# Scale area
scaler_X = StandardScaler()
X["area"] = scaler_X.fit_transform(X[["area"]])

# Scale target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.25, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

print("First 5 predicted prices: ", y_pred[:5])

# --- User input prediction ---


def get_user_input():
    # Example: you can replace input() with hardcoded values for testing
    area = float(input("Enter area: "))
    bedrooms = int(input("Enter bedrooms (1-6): "))
    bathrooms = int(input("Enter bathrooms: "))
    stories = int(input("Enter stories: "))
    mainroad = input("Mainroad (yes/no): ").strip().lower()
    guestroom = input("Guestroom (yes/no): ").strip().lower()
    basement = input("Basement (yes/no): ").strip().lower()
    hotwaterheating = input("Hotwaterheating (yes/no): ").strip().lower()
    airconditioning = input("Airconditioning (yes/no): ").strip().lower()
    parking = int(input("Parking (0-3): "))
    prefarea = input("Prefarea (yes/no): ").strip().lower()

    # Encode categorical
    mainroad = 1 if mainroad == "yes" else 0
    guestroom = 1 if guestroom == "yes" else 0
    basement = 1 if basement == "yes" else 0
    hotwaterheating = 1 if hotwaterheating == "yes" else 0
    airconditioning = 1 if airconditioning == "yes" else 0
    prefarea = 1 if prefarea == "yes" else 0

    # One-hot for bedrooms and parking
    bedrooms_oh = [0]*6
    if 1 <= bedrooms <= 6:
        bedrooms_oh[bedrooms-1] = 1
    parking_oh = [0]*4
    if 0 <= parking <= 3:
        parking_oh[parking] = 1

    # Scale area
    area_scaled = scaler_X.transform([[area]])[0][0]

    # Build input vector (order must match X columns)
    input_vector = [
        area_scaled, bedrooms, bathrooms, stories, mainroad, guestroom,
        basement, hotwaterheating, airconditioning, parking, prefarea
    ] + bedrooms_oh + parking_oh

    return input_vector


# Predict price for user input
user_input = get_user_input()
pred_scaled = model.predict([user_input])[0]
pred_price = scaler_y.inverse_transform([[pred_scaled]])[0][0]
print("Predicted house price:", int(pred_price))
