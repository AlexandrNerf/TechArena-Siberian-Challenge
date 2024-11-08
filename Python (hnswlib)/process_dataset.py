import numpy as np

import os
import sys
import argparse
import pickle

sys.path.append('hnswlibme')  # Добавляем путь к библиотеке

from hnswlibme import hnswlib
from timeit import default_timer

def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        # В начале каждого вектора записано его количество элементов
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32).reshape(-1, d + 1)
        return data[:, 1:]


def process_fvecs(input_file):

    start = default_timer()

    # Чтение векторов из исходного файла

    vectors = read_fvecs(input_file)
    print("Reading speed:", default_timer() - start)
    
    n = len(vectors)
    dim = len(vectors[0])
    start = default_timer()
    

    # Разделим датасет на 5 частей (чтобы удалять данные при их заполнении)

    data1 = vectors[:n // 5]
    data2 = vectors[n // 5:n*2//5]
    data3 = vectors[n*2 // 5:n*3//5]
    data4 = vectors[n*3 // 5:n*4//5]
    data5 = vectors[n*4 // 5:]
    del vectors

    # Выберем гиперпараметры

    m = 42 if dim < 300 else 24
    ef_const = 450 if dim < 300 else 300

    model = hnswlib.Index(space='l2', dim=dim)
    model.init_index(max_elements=n, ef_construction=ef_const, M=m)
    model.set_num_threads(4)

    # Замер скорости построения модели

    start = default_timer()
    model.add_items(data1)
    print("del data 1")
    del data1
    model.add_items(data2)
    print("del data 2")
    del data2
    model.add_items(data3)
    print("del data 3")
    del data3
    model.add_items(data4)
    print("del data 4")
    del data4
    model.add_items(data5)
    del data5
    print("done!")

    # Сохраним модель.
    
    model.save_index("model.bin")
    
    print(f"HNSW per {(default_timer() - start)} s.")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обработка fvecs')
    parser.add_argument('--data', default='gist/gist_base.fvecs', required=False, help='Папка для чтения данных')
    
    args = parser.parse_args()
    
    process_fvecs(args.data)
