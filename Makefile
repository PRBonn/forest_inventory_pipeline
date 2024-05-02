.PHONY: install
install:
	pip install -v ".[test]"

.PHONY: editable
editable:
	pip install -v --no-build-isolation -e ".[test]"

.PHONY: cpp
cpp:
	cmake -G Ninja -S . -B cpp_build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=On
	cmake --build cpp_build

.PHONY: clean
clean:
	rm -rf build
	rm -rf cpp_build
	pip uninstall forest_inventory_pipeline
