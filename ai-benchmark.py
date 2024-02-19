from ai_benchmark import AIBenchmark
import numpy as np
import warnings
np.warnings = warnings

benchmark = AIBenchmark(use_CPU=None, verbose_level=2)
results = benchmark.run(precision="normal")
