#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "hnswlib.h"


//std::vector<float> parse_query(const std::string& query_str) {
//    std::vector<float> query;
//    std::string num;
//    std::stringstream ss(query_str.substr(8)); // ������� ������� "--query="
//
//    while (std::getline(ss, num, ',')) {
//        query.push_back(std::stof(num));
//    }
//
//    return query;
//}

std::vector<float> parse_query(const std::string& query_str) {
    std::vector<float> query;

    // ������� ������� "--query=" � ������
    std::string raw_query = query_str.substr(8); // ������� ������� "--query="

    if (raw_query.front() == '[' && raw_query.back() == ']') {
        raw_query = raw_query.substr(1, raw_query.size() - 2); // ������� ������
    }
    else {
        throw std::runtime_error("Query must be in format --query=[value1,value2,...]");
    }

    // ������� �����, ����������� ��������
    std::string num;
    std::stringstream ss(raw_query);

    while (std::getline(ss, num, ',')) {
        query.push_back(std::stof(num));
    }

    return query;
}

int main(int argc, char** argv) {
    // �������� ������� ��������� --query
    if (argc < 2 || std::string(argv[1]).substr(0, 8) != "--query=") {
        std::cerr << "Usage: " << argv[0] << " --query=[comma,separated,floats]\n";
        return -1;
    }

    // ������� query
    std::vector<float> query = parse_query(argv[1]); /*{ 1.0, 1.0, 0.4, 0.2, 23.0, 454.2, 2.4, 1.2, 0.9, 0.0 };*/ 

    // �������� ������������ query
    if (query.empty()) {
        std::cerr << "Query is empty or invalid.\n";
        return -1;
    }

    // 1. �������� �������
    std::string index_filename = "hnsw_index.bin";
    int dim = query.size(); // ����������� ������� ������ ��������� � ������������ ��������
    hnswlib::SpaceInterface<float>* space = new hnswlib::L2Space(dim);

    // ������������� �������
    hnswlib::HierarchicalNSW<float> appr_alg(space, index_filename, false);

    // 2. ����� ��������� �������
    int k = 10; // ���� 10 ��������� �������
    auto result = appr_alg.searchKnn(query.data(), k);

    // 3. ����� �����������
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
