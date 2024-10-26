import numpy as np

def create_example_fvecs(file_path, num_vectors=100, vector_size=10):
    with open(file_path, 'wb') as f:
        for i in range(num_vectors):
            vec_size = np.array([vector_size], dtype=np.int32)
            vec_size.tofile(f)
            # Затем записываем сам вектор (например, от 1.0 до 7.0)
            vector = np.random.rand(vector_size).astype('float32')
            vector.tofile(f)

# Пример использования:
output_file = 'data.fvecs'
create_example_fvecs(output_file)

