#include <iostream>
#include <fstream>
#include <vector>
#include "hnswlib.h"

// Функция для чтения векторов из fvecs файла
std::vector<std::vector<float>> read_fvecs(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::vector<float>> data;
    while (!file.eof()) {
        int dim; // Первая переменная - это размерность вектора
        file.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (file.eof()) break; // Проверка конца файла после чтения

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));

        data.push_back(vec);
    }

    return data;
}

std::string parse_data_argument(int argc, char** argv) {
    std::string data_prefix = "--data=";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.substr(0, data_prefix.size()) == data_prefix) {
            return arg.substr(data_prefix.size());
        }
    }
    throw std::runtime_error("Usage: --data=path/to/data.fvecs");
}

int main(int argc, char** argv) {
    // 1. Чтение данных из файла .fvecs
    //std::string filename = "data.fvecs";
    std::string filename = parse_data_argument(argc, argv);
    std::vector<std::vector<float>> data = read_fvecs(filename);

    // if (data.empty()) {
    //     std::cerr << "Failed to read data from " << filename << std::endl;
    //     return -1;
    // }

    // 2. Инициализация hnswlib
    int dim = data[0].size(); // Размерность векторов
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dim);
    int max_elements = data.size(); // Максимальное количество элементов

    // Параметры индекса (пример: 16 связей на элемент, размер кандидатов = 200)
    hnswlib::HierarchicalNSW<float> appr_alg(space, max_elements, 16, 200);

    // 3. Добавление векторов в индекс
    for (size_t i = 0; i < data.size(); i++) {
        appr_alg.addPoint(data[i].data(), i);
    }
    
    // 4. Сохранение индекс файла
    std::string index_filename = "hnsw_index.bin";
    appr_alg.saveIndex(index_filename);
    std::cout << "HNSW index saved to " << index_filename << std::endl;

    return 0;
}
