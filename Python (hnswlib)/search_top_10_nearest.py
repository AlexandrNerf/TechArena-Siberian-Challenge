
import numpy as np
import pickle
import time
import argparse
import sys
import os
import logging
from timeit import default_timer
from contextlib import contextmanager
sys.path.append('hnswlibme') 

from hnswlibme import hnswlib


def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32).reshape(-1, d + 1)
        return data[:, 1:]

def read_ivecs(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()



def process_faiss(q):
    # Вытащим из arg вектор query

    query_list = q.strip('[]').split(',')
    query = np.array([np.float32(x) for x in query_list])
    dim = len(query)

    # Загрузка модели

    model = hnswlib.Index(space='l2', dim=dim)
    model.load_index("model.bin")

    eff = 36 if dim < 300 else 150

    model.set_ef(eff)
    
    idx, _ = model.knn_query(query, k=10)
    for i, ind in enumerate(idx[0]):
        print(f"{ind}", end='')
        if i < len(idx[0])-1:
            print(",", end='')
        else:
            print("")

    #print(f"HNSW search {(default_timer() - start)} s.")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обработка fvecs')
    parser.add_argument('--query', type=str, required=False, help='Запрос по индексам')
    
    args = parser.parse_args()
    
    process_faiss(args.query)
