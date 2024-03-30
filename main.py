import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import regularizers

# Raw Data
csv_paths_Tx = [
    '/content/drive/MyDrive/TX_lambda1.csv',
    '/content/drive/MyDrive/TX_lambda2.csv',
    '/content/drive/MyDrive/TX_lambda3.csv',
    '/content/drive/MyDrive/TX_lambda4.csv'
]

csv_paths_Rx = [
    '/content/drive/MyDrive/RX_lambda1.csv',
    '/content/drive/MyDrive/RX_lambda2.csv',
    '/content/drive/MyDrive/RX_lambda3.csv',
    '/content/drive/MyDrive/RX_lambda4.csv'
]

# Given parameters
sample_rate = 400e9  # Sample rate in samples per second
symbol_rate = 25e9  # Symbol rate in symbols per second
num_symbols = 16384  # Number of symbols
symbol_period = 1 / symbol_rate
samples_per_symbol = int(symbol_period * sample_rate)

X_Tx = read_csv_data_iterative(csv_paths_Tx, num_symbols)
X_Rx = read_csv_data_iterative(csv_paths_Rx, num_symbols)

# Use X_Tx as labels
y_labels = X_Tx

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_Rx, y_labels, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for Conv1D input
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Define the decimation factors
decimation_factors = [16, 8, 4, 2]

# Initialize lists to store accuracy values
train_accs = []
val_accs = []
train_losses = []
val_losses = []

for dec_factor in decimation_factors:
    # Decimate the training and testing data
    X_train_decimated = X_train[:, ::dec_factor]
    X_test_decimated = X_test[:, ::dec_factor]

    # Reshape the decimated data for Conv1D input
    X_train_decimated = np.expand_dims(X_train_decimated, axis=2)
    X_test_decimated = np.expand_dims(X_test_decimated, axis=2)

    # Create the model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_symbols // dec_factor, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_symbols, activation='sigmoid'))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)  # Adjust learning rate as needed
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Adjust patience value as needed

    # Train the model
    history = model.fit(X_train_decimated, y_train, epochs=100, batch_size=32, validation_data=(X_test_decimated, y_test), callbacks=[early_stopping], verbose=0)

    # Calculate accuracy values and append to the lists
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = 1 - np.mean(train_loss)
    val_acc = 1 - np.mean(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Print the training and validation loss, and accuracy for each epoch
    for epoch in range(len(train_loss)):
        print("Epoch {}: Training Loss = {:.6f}, Validation Loss = {:.6f}, Training Accuracy = {:.2%}, Validation Accuracy = {:.2%}".format(epoch+1, train_loss[epoch], val_loss[epoch], train_acc, val_acc))

# Plot the training and validation losses
for i, dec_factor in enumerate(decimation_factors):
    plt.plot(train_losses[i], label=f'Train (Decimation: {dec_factor})')
    plt.plot(val_losses[i], label=f'Validation (Decimation: {dec_factor})')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses for Different Decimation Factors')
plt.legend()
plt.show()

# Plot the training and validation accuracies
plt.figure()
for i, dec_factor in enumerate(decimation_factors):
    plt.plot(train_accs[i], label=f'Train (Decimation: {dec_factor})')
    plt.plot(val_accs[i], label=f'Validation (Decimation: {dec_factor})')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies for Different Decimation Factors')
plt.legend()
plt.show()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Adjust patience value as needed

# Initialize lists to store accuracy values
train_accs = []
val_accs = []
train_losses = []
val_losses = []

# Define the number of bits for quantization
num_bits = [2, 4, 6]

quantized_train_labels = []
quantized_test_labels = []

# Perform quantization for each number of bits
for bits in num_bits:
    model = create_model()
    quantization_accs_train = []
    quantization_accs_val = []
    # Calculate the quantization range based on the number of bits
    max_value = np.max(y_labels)
    min_value = np.min(y_labels)
    quantization_range = max_value - min_value

    # Calculate the step size for quantization
    step_size = quantization_range / (2 ** bits)

    # Quantize the labels
    quantized_train_labels_bits = np.round((y_train - min_value) / step_size) * step_size + min_value
    quantized_test_labels_bits = np.round((y_test - min_value) / step_size) * step_size + min_value

    quantized_train_labels.append(quantized_train_labels_bits)
    quantized_test_labels.append(quantized_test_labels_bits)

    for epoch in range(100):
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

        # Calculate accuracy values and append to the lists
        train_loss = history.history['loss'][0]
        train_acc = 1 - train_loss / np.mean(y_train**2)
        train_accs.append(train_acc)
        val_loss = history.history['val_loss'][0]
        val_acc = 1 - val_loss / np.mean(y_test**2)
        val_accs.append(val_acc)

        # Append quantization accuracy values to the lists
        quantization_accs_train.append(train_acc)
        quantization_accs_val.append(val_acc)

        # Append loss values to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("Epoch {}: Training Loss = {:.6f}, Validation Loss = {:.6f} Epoch {}: Training Accuracy = {:.2%}, Validation Accuracy = {:.2%}".format(epoch+1, train_loss, val_loss, epoch+1, train_acc, val_acc))

    # Plotting the accuracy comparison for the current number of bits
    plt.plot(range(1, 101), quantization_accs_train, 'b', label='Training Accuracy')
    plt.plot(range(1, 101), quantization_accs_val, 'r', label='Validation Accuracy')
    plt.title('Quantization Accuracy Comparison for {} bits'.format(bits))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

epochs = range(1, len(train_accs) + 1)

# Convert accuracy values to percentages
train_accs_percent = [acc * 100 for acc in train_accs]
val_accs_percent = [acc * 100 for acc in val_accs]

# Plotting the accuracy as percentages
plt.plot(epochs, train_accs_percent, 'b', label='Training Accuracy')
plt.plot(epochs, val_accs_percent, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Plotting the loss
plt.plot(epochs, train_losses, 'b', label='Training Loss')
plt.plot(epochs, val_losses, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

receptive_field_sizes = [3, 5, 7, 9, 11]  # Define the receptive field sizes to evaluate
accuracies = []

for receptive_field_size in receptive_field_sizes:
    model = create_model_RF(receptive_field_size)  # Pass the receptive field size to the model creation function
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=1)
    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    accuracies.append(accuracy)

# Plot the accuracy as it relates to the receptive field size
plt.plot(receptive_field_sizes, accuracies, marker='o')
plt.xlabel('Receptive Field Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Receptive Field Size')
plt.show()

X_restored = model.predict(X_test)
X_restored = scaler.inverse_transform(X_restored.reshape(X_restored.shape[0], X_restored.shape[1]))

channel_index = 0

X_Rx_scaled = X_Rx.reshape(X_Rx.shape[0], X_Rx.shape[1], 1)

# Reshape X_Rx_scaled to a 2-dimensional array
X_Rx_scaled_2d = X_Rx_scaled.reshape(X_Rx_scaled.shape[0], X_Rx_scaled.shape[1])

# Rescale the data using the same scaler used during training
X_Rx_rescaled = scaler.transform(X_Rx_scaled_2d)

# Reshape X_Rx_rescaled back to the original shape
X_Rx_rescaled = X_Rx_rescaled.reshape(X_Rx_scaled.shape[0], X_Rx_scaled.shape[1], 1)

# Perform inverse transformation to obtain the restored data
X_restored = model.predict(X_Rx_rescaled)
X_restored = scaler.inverse_transform(X_restored.reshape(X_restored.shape[0], X_restored.shape[1]))

# Generate eye diagram for transmit signal
fig, axs = plt.subplots(1, len(X_Tx[0]), figsize=(12, 4))
fig.suptitle('Eye Diagram - Transmit Signal')

symbol_limit = 1  # Set the desired number of symbols to display
batch_size = 2  # Set the batch size for processing

for channel_index in range(len(X_Tx[0])):
    ax = axs[channel_index]

    for i in range(0, min(len(X_Tx) - samples_per_symbol, symbol_limit * samples_per_symbol), samples_per_symbol):
        batch_start = i
        batch_end = min(i + samples_per_symbol * batch_size, len(X_Tx) - samples_per_symbol)

        for j in range(batch_start, batch_end, samples_per_symbol):
            ax.plot(range(samples_per_symbol), X_Tx[j:j+samples_per_symbol, channel_index], 'b-', alpha=0.5)

    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Channel {channel_index + 1}')

plt.tight_layout()
plt.show()

# Generate eye diagram for CNN model output
fig, axs = plt.subplots(1, len(X_restored[0]), figsize=(12, 4))
fig.suptitle('Eye Diagram - CNN Model Output')

for channel_index in range(len(X_restored[0])):
    ax = axs[channel_index]

    for i in range(0, min(len(X_restored) - samples_per_symbol, symbol_limit * samples_per_symbol), samples_per_symbol):
        batch_start = i
        batch_end = min(i + samples_per_symbol * batch_size, len(X_restored) - samples_per_symbol)

        for j in range(batch_start, batch_end, samples_per_symbol):
            ax.plot(range(samples_per_symbol), X_restored[j:j+samples_per_symbol, channel_index], 'b-', alpha=0.5)

    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Channel {channel_index + 1}')

plt.tight_layout()
plt.show()
