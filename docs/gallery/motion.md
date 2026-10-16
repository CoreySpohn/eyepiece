# Motion and cameras

An explanatory animation is a sequence of moves: the viewer reads while
nothing moves, then something moves while no new text arrives. This page
builds the three tools that kind of animation needs on one synthetic
coronagraph field. A `Timeline` decides when each move happens. A camera
decides where the viewer looks, either by changing an axes' limits or by
moving over a finished figure. A morph changes how the data are drawn while
every pixel keeps its identity, so the viewer can follow it from one
representation into the next.

The examples are split into six catalogs by theme. Each catalog runs in its
own kernel and includes its setup, so it can be read and executed on its own.

```{toctree}
:maxdepth: 2

motion-clock
motion-camera
motion-attention
motion-staging
motion-equations
motion-checks
```
