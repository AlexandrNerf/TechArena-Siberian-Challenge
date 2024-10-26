build: build_scan build_search
build_scan:
	g++ -O2 -I./hnswlib process_dataset.cpp -o process_dataset 
build_search:
	g++ -O2 -I./hnswlib search_top_10_nearest.cpp -o search_top_10_nearest
do_index:
	./process_dataset ${INDEX_ARGS}
do_scan:
	./search_top_10_nearest ${SCAN_ARGS}