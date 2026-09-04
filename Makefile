# Thin wrapper over the CMake pipeline. Configures on first use.
#   make            build everything
#   make verify     exhaustive MPFR check (pass = zero discrepancies)
#   make bench      build + run throughput benchmarks
#   make round      regenerate 16-bit-rounded table reports
#   make limb-tables regenerate the bf16 limb tables (ln, exp, sin)
#   make limb-sweep  measure the bf16 limb-configuration frontier
#   make glibc      fetch + extract the glibc reference source
#   make help       list every available target

BUILD_DIR ?= build
CMAKE     ?= cmake
JOBS      ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.DEFAULT_GOAL := all
.PHONY: all configure verify bench round glibc tables check-mpfr test clean distclean help \
        limb-tables limb-sweep verify-limb

$(BUILD_DIR)/CMakeCache.txt:
	$(CMAKE) -S . -B $(BUILD_DIR)

configure: $(BUILD_DIR)/CMakeCache.txt

all: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

verify bench round glibc tables check-mpfr limb-tables limb-sweep verify-limb: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS) --target $@

test: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)
	cd $(BUILD_DIR) && ctest --output-on-failure

# Removes build artifacts only; generated reports and the glibc tree stay put.
clean:
	@[ -d $(BUILD_DIR) ] && $(CMAKE) --build $(BUILD_DIR) --target clean || true

distclean:
	$(RM) -r $(BUILD_DIR)

help: configure
	@$(CMAKE) --build $(BUILD_DIR) --target help
