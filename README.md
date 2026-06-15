-changgexiangyan/resource is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
-You can click the Preview link to take a look at your changes.
---->
+# Data-Driven Oil--Water Resource Forecasting
+
+This repository contains a complete, reproducible EI-conference style research package for the topic **"Applications of machine learning and data-driven methods in oil--water resource problems"**.
+
+## Contents
+
+- `src/oil_water_ml_workflow.py` — dependency-free Python workflow that builds a heterogeneous Buckley--Leverett waterflood benchmark, trains interpretable surrogates, and exports paper-ready results.
+- `data/synthetic_waterflood_cases.csv` — generated benchmark cases.
+- `results/metrics.csv` — model accuracy table.
+- `figures/*.svg` — vector figures used by the manuscript.
+- `paper/main.tex` — full English EI-style conference paper with figures, table, and recent references.
+
+## Quick start
+
+```bash
+python src/oil_water_ml_workflow.py
+```
+
+The script uses only the Python standard library, so it runs in minimal conference-review or classroom environments without installing packages.
+
+## Main result
+
+The benchmark demonstrates that physics-feature ridge regression reaches high test-set skill for breakthrough prediction while preserving transparent coefficients and reproducibility. The workflow is intentionally lightweight so authors can replace the synthetic generator with confidential field or laboratory data without changing the evaluation protocol.
