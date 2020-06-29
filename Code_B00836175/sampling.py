import random
import numpy as np


def get_uniformly_sampled_data(data, t):
    row_count = data.shape[0]
    # Uniform sampling without replacement
    random_indices = np.random.choice(row_count, t, replace=False)
    data_t = data[random_indices, :]

    return data_t


def uniform_sampling(X_train, y_train, t):
    print(X_train[:5], "\n\n", y_train[:5])
    train = np.concatenate([X_train, y_train], axis=1)
    print("\n\n", train[:5])

    # Uniform sampling
    train_t = get_uniformly_sampled_data(train, t)

    return train_t[:, 0:-1], train_t[:, -1]


def get_oversampled_data(train_rpl, train_label, rows_per_label, row_count):
    left_rows = rows_per_label - row_count

    while (left_rows > 0):
        if left_rows <= row_count:
            train_os = get_uniformly_sampled_data(train_label, left_rows)
        else:
            train_os = get_uniformly_sampled_data(train_label, row_count)

        train_rpl = np.append(train_rpl, train_os, axis=0)

        left_rows -= row_count

    print("Final sampled shape: ", train_rpl.shape)

    return train_rpl


def balanced_uniform_sampling(X_train, y_train, t):
    print(X_train[:5], "\n\n", y_train[:5])
    train = np.concatenate([X_train, y_train], axis=1)
    print("\n\n", train[:5])

    # Balanced uniform sampling
    unique_labels = set(y_train.tolist())
    rows_per_label = int(t/len(unique_labels))
    i = 0

    for label in unique_labels:
        train_label = np.where(train[:, -1] == label)[0]
        row_count = train_label.shape[0]

        if rows_per_label <= row_count:
            # Sample uniformly
            train_rpl = get_uniformly_sampled_data(train_label, rows_per_label)
        else:
            # Sample uniformly and oversample
            train_rpl = get_uniformly_sampled_data(train_label, row_count)
            train_rpl = get_oversampled_data(
                train_rpl, train_label, rows_per_label, row_count)

        if i:
            train_t = np.append(train_t, train_rpl)
        else:
            train_t = train_rpl.copy()

        i += 1

    print("Balanced training data: ", train_t.shape)

    return train_t[:, 0:-1], train_t[:, -1]
