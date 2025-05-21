SHELL := /bin/bash

.PHONY: install clean full_rebuild partial_rebuild partial_rebuild_cpp_h partial_rebuild_py rebuild_data_utils

# Define folders and files
NNCODEC_FORK_DIR := nncodec-fork
SRC_DIR := src
DATA_UTILS_DIR := data_utils_pkg/src

# File patterns to track changes
DATA_UTILS_SRC_PY_FILES := $(shell find $(SRC_DIR) -type f -name "*.py")
NNCODEC_TRACK_FILES := $(shell find $(NNCODEC_FORK_DIR)/nnc $(NNCODEC_FORK_DIR)/nnc_core $(NNCODEC_FORK_DIR)/extensions -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" \))
SRC_CPP_H_FILES := $(shell find $(SRC_DIR) -type f \( -name "*.cpp" -o -name "*.h" \))
SRC_PY_FILES := $(shell find $(SRC_DIR) -type f -name "*.py")

# Files to keep track of rebuilds
NNCODEC_REBUILD_TRACKER := .nncodec_rebuild
SRC_REBUILD_TRACKER := .src_rebuild
SRC_CPP_H_REBUILD_TRACKER := .src_cpp_h_rebuild
SRC_PY_REBUILD_TRACKER := .src_py_rebuild
SRC_DATA_UTILS_TRACKER := .data_utils_rebuild

# Default target
install: check_full_rebuild check_partial_rebuild check_data_utils_rebuild

# Check if a full rebuild is needed
check_full_rebuild:
	@if [ ! -f $(NNCODEC_REBUILD_TRACKER) ] || [ -n "$$(find $(NNCODEC_FORK_DIR)/nnc $(NNCODEC_FORK_DIR)/nnc_core $(NNCODEC_FORK_DIR)/extensions -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.py' \) -newer $(NNCODEC_REBUILD_TRACKER))" ]; then \
		$(MAKE) full_rebuild; \
	fi

# Check if a partial rebuild is needed
check_partial_rebuild:
	@if [ ! -f $(SRC_CPP_H_REBUILD_TRACKER) ] || [ -n "$$(find $(SRC_DIR) -type f \( -name '*.cpp' -o -name '*.h' \) -newer $(SRC_CPP_H_REBUILD_TRACKER))" ]; then \
		$(MAKE) partial_rebuild_cpp_h; \
	elif [ ! -f $(SRC_PY_REBUILD_TRACKER) ] || [ -n "$$(find $(SRC_DIR) -type f -name '*.py' -newer $(SRC_PY_REBUILD_TRACKER))" ]; then \
		$(MAKE) partial_rebuild_py; \
	fi

check_data_utils_rebuild:
	@if [ ! -f $(SRC_DATA_UTILS_TRACKER) ] || [ -n "$$(find $(DATA_UTILS_DIR) -type f -name '*.py' -newer $(SRC_DATA_UTILS_TRACKER))" ]; then \
		$(MAKE) rebuild_data_utils; \
	fi

# Perform a full rebuild
full_rebuild: clean
	@echo "Performing full rebuild..."
	rm -rf $(NNCODEC_FORK_DIR)/build
	rm -rf build
	cd $(NNCODEC_FORK_DIR) && pip install .
	pip install .
	touch $(NNCODEC_REBUILD_TRACKER)
	touch $(SRC_CPP_H_REBUILD_TRACKER)
	touch $(SRC_PY_REBUILD_TRACKER)

# Perform a partial rebuild for .cpp and .h files
partial_rebuild_cpp_h:
	@echo "Performing partial rebuild for .cpp and .h files..."
	rm -rf build
	pip install .
	touch $(SRC_CPP_H_REBUILD_TRACKER)
	touch $(SRC_PY_REBUILD_TRACKER)

# Perform a partial rebuild for .py files only
partial_rebuild_py:
	@echo "Performing partial rebuild for .py files..."
	pip install .
	touch $(SRC_PY_REBUILD_TRACKER)

rebuild_data_utils:
	@echo "Rebuilding data_utils..."
	cd $(DATA_UTILS_DIR) && cd .. && pip install .
	touch $(SRC_DATA_UTILS_TRACKER)

# Clean up build directories
clean:
	@echo "Cleaning build directories..."
	rm -rf $(NNCODEC_FORK_DIR)/build
	rm -rf build
