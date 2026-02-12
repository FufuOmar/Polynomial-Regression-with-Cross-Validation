import numpy as np
from sklearn.preprocessing import StandardScaler

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
    for i in range(FOLDS):
        #Normalize, Generate polynomial features, Normalize polynomial feature ...
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
        start = end


