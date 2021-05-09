rm -rf build
bash compile.sh all ON OFF OFF ${LIB_DIR}
bash bin/run_models_benchmark.sh "static" "cpu" "1" "1"
# DNNL_VERBOSE=1 bash bin/run_models_benchmark.sh "static" "cpu" "1" "1"

