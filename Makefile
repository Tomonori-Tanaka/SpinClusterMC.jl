.PHONY: test jet

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

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