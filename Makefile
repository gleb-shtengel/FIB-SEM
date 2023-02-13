.PHONY: clean-docs
clean-docs:
	rm -rf docs/SIFT_gs

.PHONY: docs
docs: clean-docs
	pdoc3 --html --output-dir docs/ SIFT_gs
