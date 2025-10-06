"""Some helper functions for project 1."""

import csv
import numpy as np
import os
import matplotlib.pyplot as plt



def load_csv_data(data_path, sub_sample=False, keep_cols=None):
    """
    Load CSV data and return:
      - dict of training features (feature_name -> np.array)
      - dict of test features (feature_name -> np.array)
      - y_train labels
      - train_ids
      - test_ids
    """
    # --- Read header (for feature names) ---
    with open(os.path.join(data_path, "x_train.csv"), "r") as f:
        header = f.readline().strip().split(",")
    feature_names = header[1:]  # drop the first column ("Id")

    # --- Load arrays ---
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(int)
    test_ids = x_test[:, 0].astype(int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # --- Keep only selected columns (if specified) ---
    if keep_cols is not None:
        x_train = x_train[:, keep_cols]
        x_test = x_test[:, keep_cols]
        feature_names = [feature_names[i] for i in keep_cols]

    # --- Sub-sample ---
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    # --- Build dictionaries ---
    train_dict = {name: x_train[:, i] for i, name in enumerate(feature_names)}
    test_dict = {name: x_test[:, i] for i, name in enumerate(feature_names)}

    return train_dict, test_dict, y_train, train_ids, test_ids

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def distribution_plotter(y_train, train_dict):

    # count how many samples in each class
    n_pos = np.sum(y_train == 1)
    idx_pos = np.where(y_train == 1)[0]
    idx_neg = np.where(y_train == -1)[0]

    # random subsample negatives
    np.random.seed(42)
    idx_neg_sampled = np.random.choice(idx_neg, size=n_pos, replace=False)

    # number of features
    n_features = len(train_dict)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))

    for i, (feature_name, values) in enumerate(train_dict.items()):
        ax = axes.flat[i]
    
        vals_pos = values[idx_pos]
        vals_neg = values[idx_neg_sampled]
    
        ax.hist(vals_pos, bins='auto', alpha=0.5, label="y = 1")
        ax.hist(vals_neg, bins='auto', alpha=0.5, label="y = -1 (sampled)")
    
        ax.set_title(feature_name)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Count")
        ax.legend()

    # Hide empty subplots (if number of features not multiple of 4)
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes.flat[j])

    plt.tight_layout()
    plt.show()



