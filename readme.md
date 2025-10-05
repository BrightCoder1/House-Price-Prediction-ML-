# House Price Prediction - Main.py Documentation

## Overview

`Main.py` is a Python script that predicts house prices using a linear regression model. It processes the dataset `Housing.csv`, encodes categorical variables, scales features, trains a model, and allows users to input house features to get a price prediction.

---

## How It Works

### 1. Data Loading

- Reads the dataset from `Housing.csv` using pandas.

### 2. Data Preprocessing

- **Label Encoding:**  
  Converts categorical columns (`guestroom`, `mainroad`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) from 'yes'/'no' to 1/0 using `LabelEncoder`.
- **One-Hot Encoding:**  
  - `bedrooms` and `parking` columns are one-hot encoded to handle them as categorical features.
- **Feature Scaling:**  
  - The `area` column is scaled using `StandardScaler`.
  - The target variable `price` is also scaled for better model performance.

### 3. Model Training

- Splits the data into training and test sets (75% train, 25% test).
- Trains a `LinearRegression` model on the processed data.

### 4. Prediction

- Predicts prices on the test set and prints the first 5 predicted prices (in original scale).
- Provides a function for user input:
  - Prompts the user for all house features.
  - Encodes and scales the input to match the training data.
  - Predicts and prints the estimated house price.

---

## Usage

1. **Run the script:**  
   ```bash
   python Main.py
   ```
2. **Follow the prompts:**  
   Enter the required house features when prompted.

---

## Example User Input

```
Enter area: 6000
Enter bedrooms (1-6): 3
Enter bathrooms: 2
Enter stories: 2
Mainroad (yes/no): yes
Guestroom (yes/no): no
Basement (yes/no): yes
Hotwaterheating (yes/no): no
Airconditioning (yes/no): yes
Parking (0-3): 2
Prefarea (yes/no): yes
```

---

## Output

- Prints the first 5 predicted prices for the test set.
- Prints the predicted price for the user-input house.

---

## Dependencies

- pandas
- scikit-learn

Install with:
```bash
pip install pandas scikit-learn
```

---

## Notes

- Ensure `Housing.csv` is in the same directory as `Main.py`.
- The order of features in user input must match the model's expected input.

---