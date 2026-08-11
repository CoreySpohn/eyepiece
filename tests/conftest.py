"""Force a headless backend before matplotlib is imported anywhere."""

import matplotlib

matplotlib.use("Agg")
