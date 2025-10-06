import csv
import numpy as np
import os
import matplotlib.pyplot as plt

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Standard logistic regression using gradient descent.

    Args:
        y : array, shape (N,)
            Binary labels (0 or 1) for N samples.
        tx : array, shape (N, D)
            Feature matrix with N samples and D features.
        initial_w : np.ndarray, shape (D,)
            Initial weights for the model parameters.
        max_iters : int
            Number of iterations for gradient descent.
        gamma : float
            Step size for gradient descent.
    
    Returns:
        w : array, shape (D,)
            Final optimized weights.
        loss : float
            Logistic loss corresponding to the final weights.
    """
    w = initial_w
    N = y.shape[0]
    
    for _ in range(max_iters):
        z = tx @ w
        pred = 1 / (1 + np.exp(-z))
        pred = np.clip(pred, 1e-15, 1 - 1e-15)
        
        grad = tx.T @ (pred - y) / N
        w = w - gamma * grad
    
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return w, loss

def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        w: shape=(d,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    e = y - tx@w
    loss = 1/(2*N)*(np.transpose(e)@e)
    return loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        w: shape=(d, ). The vector of model parameters.

    Returns:
        An array of shape (d, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e = y - tx@w

    grad = -1/N*np.transpose(tx)@e
    return grad

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        initial_w: shape=(d, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights
        loss: final loss value 
    """
    w = initial_w
    loss = compute_loss(y, tx, w)
    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)

    return w, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        w: shape=(d, ). The vector of model parameters.

    Returns:
        An array of shape (d, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    B = y.shape[0]
    e = y - tx@w

    grad = -1/B*np.transpose(tx)@e
    return grad

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        initial_w: shape=(d, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights
        loss: final loss value 
    """

    batch_size = 1
    N = y.shape[0]
    w = initial_w
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):

        batch_indices = np.random.choice(N, batch_size, replace=False)
        y_batch = y[batch_indices]
        tx_batch = tx[batch_indices]

        grad = compute_stoch_gradient(y_batch, tx_batch, w)

        w = w - gamma * grad
        loss = compute_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """Computes the least-squares solution.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)

    Returns:
        w: shape=(d,), the optimal weights that minimize the MSE loss.
        loss : float
            MSE loss corresponding to the final weights.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss

def  ridge_regression(y,tx,lambda_):
    """Computes the ridge regression solution.

    Args:
        y: shape=(N, )
        tx: shape=(N,d)
        lamda: float

    Returns:
        w: shape=(d,), the optimal weights
        loss : float
            MSE loss corresponding to the final weights.
    """
    N = y.shape[0]
    d = tx.shape[1]

    XTX = tx.transpose()@tx
    lambda_prime = lambda_/(2*N)

    w = np.linalg.solve(XTX + lambda_prime*np.identity(d), tx.transpose()@y)
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Standard logistic regression using gradient descent.

    Args:
        y : array, shape (N,)
            Binary labels (0 or 1) for N samples.
        tx : array, shape (N, D)
            Feature matrix with N samples and D features.
        initial_w : np.ndarray, shape (D,)
            Initial weights for the model parameters.
        max_iters : int
            Number of iterations for gradient descent.
        gamma : float
            Step size for gradient descent.
    
    Returns:
        w : array, shape (D,)
            Final optimized weights.
        loss : float
            Logistic loss corresponding to the final weights.
    """
    w = initial_w
    N = y.shape[0]
    
    for _ in range(max_iters):
        z = tx @ w
        pred = 1 / (1 + np.exp(-z))
        pred = np.clip(pred, 1e-15, 1 - 1e-15)
        
        grad = tx.T @ (pred - y) / N
        w = w - gamma * grad
    
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma,sample_weights):

    """
    Regularized logistic regression (L2) using gradient descent.

    Args:
        y : array, shape (N,)
            Binary labels (0 or 1) for N samples.
        tx : array, shape (N, D)
            Feature matrix with N samples and D features.
        lambda_ : float
            Regularization parameter for L2 penalty.
        initial_w : np.ndarray, shape (D,)
            Initial weights for the model parameters.
        max_iters : int
            Number of iterations for gradient descent.
        gamma : float
            Step size for gradient descent.
    
    Returns:
        w : array, shape (D,)
            Final optimized weights.
        loss : float
            Logistic loss corresponding to the final weights (without regularization term).
    """

    w = initial_w
    N = y.shape[0]

    if sample_weights is None:
        sample_weights = np.ones(N)
    
    for _ in range(max_iters):
        z = tx @ w
        pred = 1 / (1 + np.exp(-z))
        pred = np.clip(pred, 1e-15, 1 - 1e-15)
        
        # weighted gradient
        grad = tx.T @ (sample_weights * (pred - y)) / N + lambda_ * w
        w -= gamma * grad

    # weighted loss (without regularization)
    loss = -np.mean(sample_weights * (y * np.log(pred) + (1 - y) * np.log(1 - pred)))
    #ridge_penalty = (lambda_ / 2) * np.sum(w ** 2)
    #loss = log_loss + ridge_penalty
    return w, loss

def filter_features_by_nan(train_dict, test_dict, threshold=70.0):
    """
    Drop features from dictionary if percentage of NaNs exceeds threshold.
    Returns a new dictionary with kept features.
    """
    n_rows = len(next(iter(train_dict.values())))  # number of samples
    keep = []

    for name, values in train_dict.items():
        nan_pct = np.isnan(values).sum() / n_rows * 100
        if nan_pct <= threshold:
            keep.append(name)

        train_clean_dict = {k: train_dict[k] for k in keep if k in train_dict}
        test_clean_dict = {k: test_dict[k] for k in keep if k in train_dict}

    print("Keeping", len(keep), "features")
    return train_clean_dict, test_clean_dict

def drop_and_keep_features(train_dict, test_dict, fields_to_drop):
    """
    Remove unwanted features and keep only the intersection
    of keep_names - fields_to_drop.
    """
    # Final set of features to keep

    keep_names = train_dict.keys()

    final_keep = [name for name in keep_names if name not in fields_to_drop]

    # Build cleaned dictionaries
    train_clean = {name: train_dict[name] for name in final_keep}
    test_clean  = {name: test_dict[name] for name in final_keep}

    return train_clean, test_clean


def replace_invalid_with_nan_inplace(dicts, invalid_dict):
    """
    Replace invalid values with np.nan for features listed in invalid_dict,
    applied to one or more feature dictionaries.
    
    Parameters
    ----------
    dicts : list[dict[str, np.ndarray]] or dict[str, np.ndarray]
        One or more dictionaries mapping feature name -> numpy array of values.
    invalid_dict : dict[tuple[int], list[str]]
        Mapping of invalid values -> list of feature names.
    """
    # allow single dict input
    if isinstance(dicts, dict):
        dicts = [dicts]

    for invalid_values, feature_list in invalid_dict.items():
        for name in feature_list:
            for d in dicts:
                if name in d:
                    values = d[name]
                    mask = np.isin(values, invalid_values)
                    values[mask] = np.nan
                    d[name] = values

def winsorizor_array(X, percentile=95):
    """
    Winsorizes a 2D NumPy array column-wise (each column = one feature).
    Clips values outside [100 - percentile, percentile] percentiles.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features)
    percentile : float
        Upper percentile for winsorization (e.g. 95 → clip below 5th and above 95th percentile)

    Returns
    -------
    X_winsor : np.ndarray
        Winsorized copy of X
    bounds : list of tuples
        [(lower_i, upper_i)] bounds used per feature
    """
    lower = 100 - percentile
    upper = percentile
    X_winsor = X.copy().astype(float)
    bounds = []

    for j in range(X.shape[1]):
        lo = np.nanpercentile(X[:, j], lower)
        hi = np.nanpercentile(X[:, j], upper)
        X_winsor[:, j] = np.clip(X[:, j], lo, hi)
        bounds.append((lo, hi))
    
    return X_winsor, bounds

def apply_winsor_bounds(X, bounds):
    X_clipped = X.copy().astype(float)
    for j, (lo, hi) in enumerate(bounds):
        X_clipped[:, j] = np.clip(X[:, j], lo, hi)
    return X_clipped


def categorical_nan_filler(data_dict, continuous_features):

    new_data_dict = {}

    for feature, arr in data_dict.items():
        if feature in continuous_features:
            # keep continuous as is
            new_data_dict[feature] = arr
            continue

        # convert to object for safe handling
        arr = arr.astype(object)

        # replace NaN with a string "NaN" to keep as category
        arr_safe = np.array(
            ["NaN" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
             for x in arr],
            dtype=object
        )

        # unique categories
        values = np.unique(arr_safe)

        # create one-hot columns
        for val in values:
            new_key = f"{feature}_{val}"
            new_data_dict[new_key] = (arr_safe == val).astype(int)

    return new_data_dict

def build_k_indices(y, k_fold, seed=1):

    np.random.seed(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, k_fold)
    neg_folds = np.array_split(neg_idx, k_fold)

    k_indices = [np.concatenate([pos_folds[k], neg_folds[k]]) for k in range(k_fold)]
    return k_indices


def cross_validation(y, x,x_winsor, k_fold, method, lambda_, max_iters, gamma,proba):
    k_indices = build_k_indices(y, k_fold, seed=1)
    train_losses, test_losses = [], []
    precisions, recalls, accuracies, f1s = [], [],[], []
    models = []

    for k in range(k_fold):
        test_idx = k_indices[k]
        train_idx = np.hstack([k_indices[i] for i in range(k_fold) if i != k])

        #x_train, y_train = x[train_idx], y[train_idx]
        #x_test, y_test = x[test_idx], y[test_idx]

        # Convert {-1, 1} → {0, 1}
        #y_train_lr = (y_train == 1).astype(int)
        #y_test_lr = (y_test == 1).astype(int)

        # Split data
        y_train, y_test = y[train_idx], y[test_idx]
        x_train_nonw, x_test_nonw = x[train_idx], x[test_idx]
        x_train_winsor, x_test_winsor = x_winsor[train_idx], x_winsor[test_idx]

        # === Winsorize (fit on train, apply to test) ===
        # === Apply log transform safely ===
        x_train_winsor = np.log1p(np.clip(x_train_winsor, a_min=0, a_max=None))
        x_test_winsor  = np.log1p(np.clip(x_test_winsor,  a_min=0, a_max=None))

        # Combine continuous + other features
        x_train = np.column_stack([x_train_winsor, x_train_nonw])
        x_test  = np.column_stack([x_test_winsor,  x_test_nonw])

        # Convert {-1, 1} → {0, 1}
        y_train_lr = (y_train == 1).astype(int)
        y_test_lr  = (y_test == 1).astype(int)

        if method == 'undersample':
            # old logic
            minority_idx = np.where(y_train == 1)[0]
            majority_idx = np.where(y_train == -1)[0]
            n_minority = len(minority_idx)
            n_majority = len(majority_idx)
            if n_minority == 0 or n_majority == 0:
                print(f"Fold {k+1}: one class missing, skipping undersampling.")
            else:
                sampled_majority = np.random.choice(majority_idx, size=n_minority, replace=False)
                balanced_idx = np.concatenate([minority_idx, sampled_majority])
                x_train, y_train_lr = x_train[balanced_idx], y_train_lr[balanced_idx]
                sample_weights = np.ones_like(y_train_lr, dtype=float)

        elif method == 'weighted':
            # compute weights inversely proportional to class frequency
            n_pos = np.sum(y_train_lr == 1)
            n_neg = np.sum(y_train_lr == 0)
            N = len(y_train_lr)
            w_pos = N / (2 * n_pos)
            w_neg = N / (2 * n_neg)
            sample_weights = np.where(y_train_lr == 1, w_pos, w_neg)
            sample_weights = sample_weights / np.mean(sample_weights)

        else:
            sample_weights = np.ones_like(y_train_lr, dtype=float)

        # Train model
        initial_w = np.zeros(x_train.shape[1])
        w, train_loss = reg_logistic_regression(y_train_lr, x_train, lambda_, initial_w, max_iters, gamma, sample_weights)

        # Predict on test set
        z_test = x_test @ w
        pred_prob = 1 / (1 + np.exp(-z_test))
        pred_label = (pred_prob >= proba).astype(int)

        # Loss
        test_loss = -np.mean(y_test_lr * np.log(pred_prob) + (1 - y_test_lr) * np.log(1 - pred_prob))
        #test_loss += (lambda_ / 2) * np.sum(w ** 2)

        # Metrics
        tp = np.sum((pred_label == 1) & (y_test_lr == 1))
        fp = np.sum((pred_label == 1) & (y_test_lr == 0))
        fn = np.sum((pred_label == 0) & (y_test_lr == 1))
        tn = np.sum((pred_label == 0) & (y_test_lr == 0))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)
        f1s.append(f1)
        models.append(w)

        print(f"Fold {k+1}/{k_fold} → test loss: {test_loss:.4f}, train loss:{train_loss:.4f} prec: {precision:.3f}, rec: {recall:.3f}, acc: {accuracy:.3f}, f1: {f1:.3f}")

    print("\n=== Cross-Validation Summary ===")
    print(f"Avg train loss: {np.mean(train_losses):.4f}")
    print(f"Avg test loss: {np.mean(test_losses):.4f}")
    print(f"Avg precision: {np.mean(precisions):.3f}")
    print(f"Avg recall:    {np.mean(recalls):.3f}")
    print(f"Avg accuracy:  {np.mean(accuracies):.3f}")
    print(f"Avg f1:  {np.mean(f1s):.3f}")

    return {
        "train_loss": np.mean(train_losses),
        "test_loss": np.mean(test_losses),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "accuracy": np.mean(accuracies),
        "f1": np.mean(f1s),
        "models": models
    }