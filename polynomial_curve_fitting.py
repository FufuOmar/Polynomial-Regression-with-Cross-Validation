import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import subprocess
#subprocess.run(['bash', 'learn_script'])
# ============================================
# Load the training data
# ============================================
FOLDS = 12
DEGREES = 20
train_data = np.loadtxt('train.dat')
years = train_data[:, 0]      # First column: years
debt = train_data[:, 1]       # Second column: debt %

# ============================================
# Split the Data
# ============================================
start = 0
n = len(train_data) # 60
fold_size = n // FOLDS # Whole number
for d in range(DEGREES + 1):
    print(f"Degree: {d}")
    start = 0
    remainder = n % FOLDS
    fold_train_rmses = []
    fold_test_rmses = []
    for i in range(FOLDS):
        if remainder > 0:
            current_size = fold_size + 1
            remainder -= 1
        else:
            current_size = fold_size
        end = start + current_size
        
        # This fold becomes the test set, the rest becomes the training set. Every iteration, the test set changes.
        cv_test = train_data[start:end]
        cv_train = np.concatenate(
            (train_data[:start], train_data[end:])
        )
        print(f"Fold {i+1}:\n Test:\n {cv_test}")

        # ============================================
        # Normalization
        # ============================================

        # Scales (year - avg) / std, where avg is avg of the years in the data
        scaler_x = StandardScaler()
        cv_train_years_scaled = scaler_x.fit_transform(cv_train[:, 0].reshape(-1, 1)) # Scale only the years in the training set
        cv_test_years_scaled  = scaler_x.transform(cv_test[:, 0].reshape(-1, 1)) # Scale only the years in the test set using the **same scaler**

        scaler_y = StandardScaler()
        cv_train_debt_scaled = scaler_y.fit_transform(cv_train[:, 1].reshape(-1, 1)) # Scale only the debt in the training set
        cv_test_debt_scaled  = scaler_y.transform(cv_test[:, 1].reshape(-1, 1)) # Scale only the debt in the test set using the **same scaler**
        if d == 0:
            cv_train_poly_scaled = np.ones((len(cv_train), 1))  # Only bias term
            cv_test_poly_scaled  = np.ones((len(cv_test), 1))
        else:
            poly = PolynomialFeatures(degree = d, include_bias=False)
            cv_train_poly = poly.fit_transform(cv_train_years_scaled) # Transform the scaled years into polynomial
            cv_test_poly = poly.transform(cv_test_years_scaled) # Transform the scaled years in the test set using the same polynomial features

            scaler_poly = StandardScaler()
            cv_train_poly_scaled = scaler_poly.fit_transform(cv_train_poly) # Scale the polynomial features
            cv_test_poly_scaled = scaler_poly.transform(cv_test_poly) # Scale the polynomial features in the test set using the **same scaler**

            cv_train_poly_scaled = np.hstack((np.ones((cv_train_poly_scaled.shape[0], 1)), cv_train_poly_scaled)) # Add bias term to the polynomial features in the training set
            cv_test_poly_scaled = np.hstack((np.ones((cv_test_poly_scaled.shape[0], 1)), cv_test_poly_scaled)) # Add bias term to the polynomial features in the test set

        # ============================================
        # Calculate Weights
        # ============================================
        regr = Ridge(alpha=0, fit_intercept=False, solver='cholesky')
        regr.fit(cv_train_poly_scaled, cv_train_debt_scaled)

        # ============================================
        # Predictions
        # ============================================
        cv_train_predict_scaled = regr.predict(cv_train_poly_scaled)
        cv_test_predict_scaled  = regr.predict(cv_test_poly_scaled)

        # Inverse transform back to original debt space
        cv_train_predict = scaler_y.inverse_transform(cv_train_predict_scaled.reshape(-1, 1))
        cv_test_predict  = scaler_y.inverse_transform(cv_test_predict_scaled.reshape(-1, 1))

        # Get actual debt values in original space
        cv_train_actual = cv_train[:, 1].reshape(-1, 1)
        cv_test_actual  = cv_test[:, 1].reshape(-1, 1)

        # ============================================
        # Calulate RMSE 
        # ============================================
        train_rmse = np.sqrt(np.mean((cv_train_actual - cv_train_predict) ** 2))
        test_rmse  = np.sqrt(np.mean((cv_test_actual  - cv_test_predict)  ** 2))

        fold_train_rmses.append(train_rmse)
        fold_test_rmses.append(test_rmse)

        start = end
        
    avg_train_rmse = np.mean(fold_train_rmses)
    avg_test_rmse = np.mean(fold_test_rmses)
    print(f"Average Train RMSE: {avg_train_rmse:.4f}")
    print(f"Average Test RMSE: {avg_test_rmse:.4f}\n")


