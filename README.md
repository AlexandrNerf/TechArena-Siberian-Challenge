# TechArena: Vector Search

Цель задания: написать ANN для получения лучшей метрики recall@10 и QPS на датасетах SIFT1M и GIST1M.

Выбранный алгоритм, его исследование и подбор находятся в папке Python в pdf файле.

Чуть ниже о структуре...

## C++ (Второстепенное решение)

Тестировалось, но не использовалось в качестве главного. Использовалась библиотека <b>hnswlib C++ (header-only)</b>

## Python (Основное решение)

Используемые библиотеки: <b>hnswlib</b> + <b>numpy</b>.

Подробности в файле с техническим решением.

## Benchmark 

В этой папке - замеры baseline и нашего решения по параметру Query Per Second.
