# Define the function to read and preprocess the data iteratively
def read_csv_data_iterative(csv_paths, num_symbols):
    data_points = []
    for csv_path in csv_paths:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                for value in row:
                    data_points.append(float(value))
    data_points = np.array(data_points)
    num_samples = len(data_points) // num_symbols
    data_points = data_points[:num_samples * num_symbols]
    data_points = data_points.reshape(num_samples, num_symbols)
    return data_points
