.PHONY: test test-slow jet examples-smoke

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

test-slow:
	julia --project=. -e 'using Pkg; Pkg.test(test_args=["slow"])'

# Runs the fast examples to catch breakages in their API usage. The slow
# ones (02 cooling run, 05 ferh sublattice) are skipped: 02 takes minutes
# at full sweep counts, and 05 sweeps ferh_4x4x4 which has ~840k cluster
# instances. Run them manually from a checkout if needed.
examples-smoke:
	@for f in examples/01_quickstart.jl examples/03_anisotropy_demo.jl examples/04_initial_spin_presets.jl; do \
	    echo "==> $$f"; \
	    julia --project=. "$$f" || exit 1; \
	done

# Static analysis with JET.jl.
# Runs in a temporary environment (does not modify Manifest.toml).
# Use 'make jet ARGS=--fail' to exit non-zero when reports are found (CI use).
jet:
	julia --startup-file=no scripts/dev/jet_analysis.jl $(ARGS)

#test-unit:
#	TEST_MODE=unit julia --project -e 'using Pkg; Pkg.test()'

#test-integration:
#	TEST_MODE=integration julia --project -e 'using Pkg; Pkg.test()'

#test-develop:
#	TEST_MODE=develop julia --project -e 'using Pkg; Pkg.test()'