def create_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_symbols, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(num_symbols, activation='sigmoid')) 
    optimizer = Adam(learning_rate=0.001)  # Adjust learning rate as needed
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model
