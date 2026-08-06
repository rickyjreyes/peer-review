PYTHON ?= python
RSCRIPT ?= Rscript

.PHONY: test calibrate calibrate-quick figures robustness

test:
	$(PYTHON) -m unittest discover -s tests -v

calibrate:
	$(PYTHON) src/calibrate_empirical_model.py

calibrate-quick:
	$(PYTHON) src/calibrate_empirical_model.py --quick --output-dir /tmp/peer-review-calibration-quick

figures:
	$(RSCRIPT) src/plot_results.R

robustness:
	$(PYTHON) src/reproduce_robustness_experiments.py --output-root reproduced_results
