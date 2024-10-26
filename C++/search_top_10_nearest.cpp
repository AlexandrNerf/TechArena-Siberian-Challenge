#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "hnswlib.h"


//std::vector<float> parse_query(const std::string& query_str) {
//    std::vector<float> query;
//    std::string num;
//    std::stringstream ss(query_str.substr(8)); // Убираем префикс "--query="
//
//    while (std::getline(ss, num, ',')) {
//        query.push_back(std::stof(num));
//    }
//
//    return query;
//}

std::vector<float> parse_query(const std::string& query_str) {
    std::vector<float> query;

    // Убираем префикс "--query=" и скобки
    std::string raw_query = query_str.substr(8); // Убираем префикс "--query="

    if (raw_query.front() == '[' && raw_query.back() == ']') {
        raw_query = raw_query.substr(1, raw_query.size() - 2); // Убираем скобки
    }
    else {
        throw std::runtime_error("Query must be in format --query=[value1,value2,...]");
    }

    // Парсинг чисел, разделенных запятыми
    std::string num;
    std::stringstream ss(raw_query);

    while (std::getline(ss, num, ',')) {
        query.push_back(std::stof(num));
    }

    return query;
}

int main(int argc, char** argv) {
    // Проверка наличия аргумента --query
    if (argc < 2 || std::string(argv[1]).substr(0, 8) != "--query=") {
        std::cerr << "Usage: " << argv[0] << " --query=[comma,separated,floats]\n";
        return -1;
    }

    // Парсинг query
    std::vector<float> query = parse_query(argv[1]); /*{ 1.0, 1.0, 0.4, 0.2, 23.0, 454.2, 2.4, 1.2, 0.9, 0.0 };*/ 

    // Проверка корректности query
    if (query.empty()) {
        std::cerr << "Query is empty or invalid.\n";
        return -1;
    }

    // 1. Загрузка индекса
    std::string index_filename = "hnsw_index.bin";
    int dim = query.size(); // Размерность запроса должна совпадать с размерностью векторов
    hnswlib::SpaceInterface<float>* space = new hnswlib::L2Space(dim);

    // Инициализация индекса
    hnswlib::HierarchicalNSW<float> appr_alg(space, index_filename, false);

    // 2. Поиск ближайших соседей
    int k = 10; // Ищем 10 ближайших соседей
    auto result = appr_alg.searchKnn(query.data(), k);

    // 3. Вывод результатов
    //std::cout << "Nearest neighbors for query:\n";
    while (!result.empty()) {
        auto neighbor = result.top();
        std::cout << neighbor.second;
        result.pop();
        if (!result.empty())
            std::cout << ',';
        else
            std::cout << std::endl;
    }

    return 0;
}
