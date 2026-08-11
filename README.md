# eyepiece

Plotting and animation primitives for astronomical imaging simulations.

## What eyepiece is

`eyepiece` provides ax-first, stateless figure functions for the plots that
recur across direct-imaging work: log-scaled and diverging image displays,
complex-field panels, corner plots, orbit and sky-track scenes, optical-train
schematics, and a multi-sink animation recorder that grabs frames once and
writes them to any combination of gif, mp4, and html. Every primitive takes
plain arrays in and returns a small result object holding the axes and the
artists it drew, so callers can update or extend a figure without re-deriving
which line or image object came from where.

Figure style (color, colormaps, and mode) is resolved through
[hwostyle](https://github.com/HabitableWorldsObservatory/hwostyle) at call
time, not at import time. `import eyepiece` never activates a style or
touches matplotlib's rcParams. Without an active hwostyle mode, primitives
fall back to a fixed light-mode palette, so eyepiece also works as a
standalone plotting library.

## What eyepiece is not

- **Not a simulation library.** eyepiece takes arrays and returns figures; it
  never imports a simulation, orbit, or optics package.
- **Not a style engine.** Color, colormap, and mode definitions live in
  hwostyle; eyepiece only reads them at call time.
- **Not a data pipeline.** Aggregating or transforming simulation output
  before plotting is the caller's job.

## Status

eyepiece is in early development. The package currently exposes only its
version; the plotting and animation API is under active construction.

## Installation

```bash
pip install eyepiece
```

Unit conversions used by a small number of layout helpers (arcsecond and AU
extents) are optional and pull in [hwoutils](https://github.com/CoreySpohn/hwoutils):

```bash
pip install eyepiece[hwo]
```

## License

MIT
