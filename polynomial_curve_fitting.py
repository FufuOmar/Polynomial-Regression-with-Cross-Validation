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
avg_train_rmse = np.zeros(DEGREES + 1)
avg_test_rmse = np.zeros(DEGREES + 1)

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
        
    avg_train_rmse[d] = np.mean(fold_train_rmses)
    avg_test_rmse[d] = np.mean(fold_test_rmses)
    print(f"Average Train RMSE: {avg_train_rmse[d]:.4f}")
    print(f"Average Test RMSE: {avg_test_rmse[d]:.4f}\n")

d_star = np.argmin(avg_test_rmse)
print(f"Best Degree (d*): {d_star}")

# ============================================
# Retrain on ALL training data with d* 
# ============================================
final_scaler_x = StandardScaler()
train_years_scaled = final_scaler_x.fit_transform(years.reshape(-1,1)) 
final_scaler_y = StandardScaler()
train_debt_scaled = final_scaler_y.fit_transform(debt.reshape(-1,1))

if d_star == 0:
    train_poly_scaled = np.ones((len(years), 1))
else:
    poly = PolynomialFeatures(degree=d_star, include_bias=False) 
    train_poly = poly.fit_transform(train_years_scaled)
    scaler_poly = StandardScaler()
    train_poly_scaled = scaler_poly.fit_transform(train_poly)
    train_poly_scaled = np.hstack((np.ones((train_poly_scaled.shape[0], 1)), train_poly_scaled))

# Calculate Weights
weights = Ridge(alpha=0, fit_intercept=False, solver='cholesky')
weights.fit(train_poly_scaled, train_debt_scaled)

final_train_prediction_scaled = weights.predict(train_poly_scaled)
final_train_prediction = final_scaler_y.inverse_transform(final_train_prediction_scaled.reshape(-1,1))

# Calculate RMSE
final_train_rmse = np.sqrt(np.mean((debt.reshape(-1, 1) - final_train_prediction) ** 2))

# ============================================
# Evaluate on test.dat
# ============================================
test_data = np.loadtxt('test.dat')
test_years = test_data[:, 0]
test_debt = test_data[:, 1]

test_years_scaled = final_scaler_x.transform(test_years.reshape(-1, 1))

if d_star == 0:
    test_poly_scaled = np.ones((len(test_years), 1))
else:
    test_poly = poly.transform(test_years_scaled)
    test_poly_scaled = scaler_poly.transform(test_poly)
    test_poly_scaled = np.hstack((np.ones((test_poly_scaled.shape[0], 1)), test_poly_scaled))

final_test_prediction_scaled = weights.predict(test_poly_scaled)
final_test_prediction = final_scaler_y.inverse_transform(final_test_prediction_scaled.reshape(-1, 1))

final_test_rmse = np.sqrt(np.mean((test_debt.reshape(-1, 1) - final_test_prediction) ** 2))


# ============================================
# Print Results
# ============================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Optimal Degree (d*): {d_star}")
print(f"Final Training RMSE: {final_train_rmse:.6f}")
print(f"Final Test RMSE:     {final_test_rmse:.6f}")
print("\nCoefficient Weights:")
for i, w in enumerate(weights.coef_.flatten()):
    print(f"  w{i} = {w:.18e}")

# ============================================
# Create Plot
# ============================================
import matplotlib.pyplot as plt

# Create smooth curve from 1938 to 2024
x_curve = np.linspace(1938.0, 2024.0, 500).reshape(-1, 1)
x_curve_scaled = final_scaler_x.transform(x_curve)
x_curve_poly = poly.transform(x_curve_scaled)
x_curve_poly_scaled = scaler_poly.transform(x_curve_poly)
x_curve_poly_scaled = np.hstack((np.ones((x_curve_poly_scaled.shape[0], 1)), x_curve_poly_scaled))

y_curve_scaled = weights.predict(x_curve_poly_scaled)  
y_curve = final_scaler_y.inverse_transform(y_curve_scaled.reshape(-1, 1))
y_curve = final_scaler_y.inverse_transform(y_curve_scaled.reshape(-1, 1))

# Plot
plt.figure(figsize=(12, 7))
plt.scatter(years, debt, color='blue', alpha=0.6, s=50, label='Training Data')
plt.plot(x_curve, y_curve, color='red', linewidth=2, label=f'Polynomial (degree {d_star})')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Federal Debt as % of GDP', fontsize=12)
plt.title(f'U.S. Federal Debt Prediction (Polynomial Degree {d_star})', fontsize=14)
plt.xlim(1938, 2024)
plt.ylim(0, 140) 
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('polynomial_fit.png', dpi=300)
plt.show()

print("\nPlot saved as 'polynomial_fit.png'")






