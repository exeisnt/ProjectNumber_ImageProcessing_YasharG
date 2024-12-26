# Load CIFAR-10 dataset as an example
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

# Load CIFAR-10 dataset as an example
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)
