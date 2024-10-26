
import numpy as np
import pickle
import time
import os
import argparse
import sys
from timeit import default_timer
sys.path.append('hnswlibme') 

from hnswlibme import hnswlib

# Чтение fvecs
def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32).reshape(-1, d + 1)
        return data[:, 1:]

# Чтение ivecs
def read_ivecs(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# Замеряем скорость на GIST
def benchmarking(q):

    start = default_timer()
    queries = read_fvecs("gist/gist_query.fvecs")

    dim = len(queries[0])

    model = hnswlib.Index(space='l2', dim=dim)
    model.load_index("model.bin")
    model.set_ef(24)
    
    print(f"Loaded in {default_timer() - start} s.")

    start = default_timer()
    for query in queries:
        idx, _ = model.knn_query(query, k=10)
        for i, ind in enumerate(idx[0]):
            print(f"{ind}", end='')
            if i != len(idx[0])-1:
                print(",", end='')
            else:
                print("")

    print(f"HNSW (hnswlib) QPS is { len(queries) / (default_timer() - start)}.")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обработка fvecs')
    parser.add_argument('--query', type=str, required=False, help='Запрос по индексам')
    
    args = parser.parse_args()
    
    benchmarking(args.query)
