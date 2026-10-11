# Changelog

## [0.8.0](https://github.com/CoreySpohn/eyepiece/compare/v0.7.0...v0.8.0) (2026-10-05)


### Features

* **anim:** add RawSink and record(free_limits=) for axes whose limits move ([3bf88c4](https://github.com/CoreySpohn/eyepiece/commit/3bf88c46cd37927cd02fa0be0e4625581129f08d))
* **camera:** add PageCamera, overview_box and spotlight ([5ed474f](https://github.com/CoreySpohn/eyepiece/commit/5ed474f897cb328a6e22aec03ec513a76e92d861))
* **deps:** require matplotlib 3.11 and drop the version gates ([a441d03](https://github.com/CoreySpohn/eyepiece/commit/a441d033e1fcc9877b297d15398071c2810b48b3))
* **manim:** make ensure_font and Units public ([8e84791](https://github.com/CoreySpohn/eyepiece/commit/8e84791cc0c2b35ed18e6be1237faaaf82a717c9))
* **morph:** add quad_image and brush ([9f825bb](https://github.com/CoreySpohn/eyepiece/commit/9f825bb22876712f5cb3d624e0fc12feb7e8fe4f))
* **motion:** add Timeline, Plan, ease, stagger, frame_count, zoom_path, view_limits and pixel_quads ([02693eb](https://github.com/CoreySpohn/eyepiece/commit/02693eb790e2dd7561e5832d67a391f3d5f5a9b4))

## [0.7.0](https://github.com/CoreySpohn/eyepiece/compare/v0.6.0...v0.7.0) (2026-10-03)


### Features

* **emphasis:** add fade by alpha for arrivals and step_list for narrated build-ups ([0b5299d](https://github.com/CoreySpohn/eyepiece/commit/0b5299d5ed95096e1c42ad52adf0d9b4c9364c02))
* **images:** add centers= to the image primitives, kymograph, overlay_line, and bracket ([2faa8a0](https://github.com/CoreySpohn/eyepiece/commit/2faa8a040c00c36eb521b9a0f21fe70f3d2ef0b1))
* **images:** add ruler, a dimension arrow with its label on a backing box ([13e410f](https://github.com/CoreySpohn/eyepiece/commit/13e410fd161acfc58847c364947e856ea40ce942))
* **images:** give overlay_circle a line style and a label on a backing box ([44dd250](https://github.com/CoreySpohn/eyepiece/commit/44dd25098c41a1e12f544e3f3a461508f6ba757f))
* **insets:** add curve_insets, square insets standing over marked points of a curve ([76f907e](https://github.com/CoreySpohn/eyepiece/commit/76f907e58a489ef58cd0440f08240dde68d08b96))
* **phasor:** add levels, dashed equal-brightness circles about zero with labels ([e74c4c1](https://github.com/CoreySpohn/eyepiece/commit/e74c4c15ef136414b400121287ad1389fc4bc524))
* **phasor:** add per-arrow starts, overlap separation, axes-unit dial size, axis-label text_kw, and a public phase_ring ([2336b37](https://github.com/CoreySpohn/eyepiece/commit/2336b3719ebad95afb27ad81b78bc842b908ed10))
* **phasor:** add phasor, arrows and chains on the complex plane with a phase ring and inset dials ([104d9dd](https://github.com/CoreySpohn/eyepiece/commit/104d9dd684a0a79dc5548815118bc4ba1ecbe7ec))
* **phasor:** add sum_head_scale and sum_width for the resultant ([fe1e8fc](https://github.com/CoreySpohn/eyepiece/commit/fe1e8fc6322d0a6745f0250ace88687cee8a4acc))
* **result:** add "arrow" to ARTIST_KEYS ([97323d3](https://github.com/CoreySpohn/eyepiece/commit/97323d33ec009b442f96db547600e790a13c4e2d))
* **result:** add an insets slot for axes a primitive creates beside its own ([1ad9896](https://github.com/CoreySpohn/eyepiece/commit/1ad9896816654cd0b399fbb9068542d918dea12a))
* **scene:** give fading_track an update that redraws the ramp along a new path ([fd1ce28](https://github.com/CoreySpohn/eyepiece/commit/fd1ce28f1c1674916802fe3aca6e74bd44c5c080))
* **schematic:** add data coordinates, per-glyph colors, multi-plane highlight, gids and an in-place update to rail ([bfc43c6](https://github.com/CoreySpohn/eyepiece/commit/bfc43c6610d2cf94453f639111bc16522d94b739))
* **schematic:** add rail beam, bare, optional, marker_colors, label_y, and flat_mirror, beam_splitter, field_stop glyphs ([361958a](https://github.com/CoreySpohn/eyepiece/commit/361958a6a8b3c7378a147e5472ff50e38973d736))
* **schematic:** add rail linewidth_scale and label_kw, and keep scaled marker widths through update ([4425a30](https://github.com/CoreySpohn/eyepiece/commit/4425a30c29697f09f6ee5298da888e9b863bbd38))
* **schematic:** add rail_panels, axes hung under the planes of a drawn rail ([9b943aa](https://github.com/CoreySpohn/eyepiece/commit/9b943aaae245db111d0bb563335fc4ba80dfa1b4))
* **stats:** add convergence, samples and their running mean or sum against labeled references ([9d1fc45](https://github.com/CoreySpohn/eyepiece/commit/9d1fc45dd3b2542f8e1a978358c69ddd36096947))
* **stats:** add signed_trace and hist_fill, sample-by-sample reveals about a level and into bins ([9e0e5b6](https://github.com/CoreySpohn/eyepiece/commit/9e0e5b6c6e95e8f45f28976438b7dabdf028e682))


### Bug Fixes

* **anim:** freeze the layout before the first grab so frame 0 matches every later frame ([5bc3ee7](https://github.com/CoreySpohn/eyepiece/commit/5bc3ee77b918a57f852b5efc24ec598e00e73a89))
* **anim:** settle the constrained layout at the sink dpi before freezing it ([6fa7ed0](https://github.com/CoreySpohn/eyepiece/commit/6fa7ed014e662c1b3a77f50d63e6eb3afe51f941))
* **phasor:** accept per-arrow colors and line styles that mix kinds of spec ([11325ea](https://github.com/CoreySpohn/eyepiece/commit/11325ea64bfe60978aee7ee2e02073497f388fe7))
* **phasor:** draw a zero-size head as a plain shaft instead of dividing by zero ([be7c88f](https://github.com/CoreySpohn/eyepiece/commit/be7c88fbafafca78ee09fb3d7595b51f1e27e70c))
* **phasor:** keep zero-length arrows out of the arrowhead geometry ([c03a87c](https://github.com/CoreySpohn/eyepiece/commit/c03a87c71090d80dc093e594b50eac76d86a6be3))
* **schematic:** make rail gids unique with indexed parts and occurrence-named repeated labels ([e7b4b6f](https://github.com/CoreySpohn/eyepiece/commit/e7b4b6fa3964360c9f8ff078d926a750f11421bc))
* **stats:** let convergence reference styles mix names and dash patterns ([e4d57f6](https://github.com/CoreySpohn/eyepiece/commit/e4d57f67c4a7046e840fd23904aa48db76d4ce05))

## [0.6.0](https://github.com/CoreySpohn/eyepiece/compare/v0.5.0...v0.6.0) (2026-09-29)


### Features

* **emphasis:** expose blend, the color blend fade applies, for a single color ([0de2201](https://github.com/CoreySpohn/eyepiece/commit/0de220125bc47f9ea5b6ccadb8c233559cf95ed8))
* **images:** let overlay_circle take underlay_kw to set the underlay width and tag ([58c6ffe](https://github.com/CoreySpohn/eyepiece/commit/58c6ffe3458902bfbf100d50ecdf5567c3e62e31))
* **schematic:** add dm and phase_mask glyphs, mid-gap Fourier lenses, and beam-narrowing stops to rail ([7aea900](https://github.com/CoreySpohn/eyepiece/commit/7aea900e47b3121f1969b829a3b718e9850c8e9d))

## [0.5.0](https://github.com/CoreySpohn/eyepiece/compare/v0.4.0...v0.5.0) (2026-09-29)


### Features

* **emphasis:** add fade and capture to carry earlier figure steps forward faded ([f735e07](https://github.com/CoreySpohn/eyepiece/commit/f735e077313a3d19b5678c7931b615d2e559067d))
* **images:** add overlay_circle, a dashed circle legible on bright and dark pixels ([d151356](https://github.com/CoreySpohn/eyepiece/commit/d151356ce8c5d88fc4e936a1249770f51be31252))
* **images:** compare_grid lays images sharing one norm on a 2D grid with empty cells ([8429e31](https://github.com/CoreySpohn/eyepiece/commit/8429e31319cfb1696c233c02703ca82fb509a5b1))
* **images:** let compare_row and compare_grid draw the shared colorbar into a caller cax ([c73e25d](https://github.com/CoreySpohn/eyepiece/commit/c73e25de3d221e28b2fd943aa8cfb908eb3e6a65))
* **manim:** render prepared views with replayable playback ([f02d4d2](https://github.com/CoreySpohn/eyepiece/commit/f02d4d2bf6c917d2a046b2f338036b85af2fc9d4))
* **mpl:** render prepared views and sequences ([7074e2b](https://github.com/CoreySpohn/eyepiece/commit/7074e2b81249d820ea07838c3802b7a4357f2564))
* **prepared:** add numerical views and lazy imports ([403bca9](https://github.com/CoreySpohn/eyepiece/commit/403bca9b8c3500479276b2fcb6a0c4ffa536e0f7))
* **render:** weight point sets, halo image labels, and share one font file ([fb85fab](https://github.com/CoreySpohn/eyepiece/commit/fb85fab9424e5754f25ed8cbf3b2cd8569ca1d46))
* **schematic:** let rail draw relay and lensless gaps between planes ([fd91df8](https://github.com/CoreySpohn/eyepiece/commit/fd91df88648ae4555fc07912a72e86b3b9bb2012))
* **sequence:** add replayable states and shared time sampling ([0ddce4b](https://github.com/CoreySpohn/eyepiece/commit/0ddce4b957b08d567c70aaf1d7c5ecb159714f69))
* **style:** share display mapping and render profiles ([3ae8a0a](https://github.com/CoreySpohn/eyepiece/commit/3ae8a0aab11dab08ce9226cbe7658afd6c30a77e))


### Bug Fixes

* **api:** keep schematic callable after lazy submodule loads ([ba5d012](https://github.com/CoreySpohn/eyepiece/commit/ba5d01282d6496c08c4d8f9036b82b9998cb8171))
* **api:** return submodules through lazy attribute access ([9d4b5e6](https://github.com/CoreySpohn/eyepiece/commit/9d4b5e6c5a140f38b6333b8d85f015bca13dc16b))
* **manim:** format tiny tick values compactly ([e61a10c](https://github.com/CoreySpohn/eyepiece/commit/e61a10c8cf7e1b9f0e6f5712a5260f50beedf41d))
* **manim:** guard playback timing and import purity ([7077b95](https://github.com/CoreySpohn/eyepiece/commit/7077b95385e16b4ac2612cd528af99dad94579bb))
* **manim:** redraw changing labels during playback ([4b3d985](https://github.com/CoreySpohn/eyepiece/commit/4b3d9854ab477f6800f339e022ad42c0e8b76bb4))
* **mpl:** align colorbar lookup and draw coordinate gaps ([017c959](https://github.com/CoreySpohn/eyepiece/commit/017c959c20f4cc3eb856bffdbc660ec12d87ff4e))
* **prepared:** tighten view validation ([b62e86f](https://github.com/CoreySpohn/eyepiece/commit/b62e86fb484413c3681391d6f6b8129a8faab47b))
* **prepared:** validate scales before mapping ([5f716de](https://github.com/CoreySpohn/eyepiece/commit/5f716def05e54e7d7553ed6cada42b65f41787fd))
* **render:** paper-sized defaults, outline image regions, clock formats ([649ab1f](https://github.com/CoreySpohn/eyepiece/commit/649ab1f2b6c2dacec4fd88019014da5adb15b5bb))
* **sequence:** compose mark channels and snap sample times ([3a4ba13](https://github.com/CoreySpohn/eyepiece/commit/3a4ba13d4be249f34aafd5ab88319cebd08f088c))
* **sequence:** label each strip slot once, through its Clock when it has one ([97d06e4](https://github.com/CoreySpohn/eyepiece/commit/97d06e452266f15cac6ee8fb07597a65ef2b2ed2))
* **sequence:** label nested strip panels ([38a3a9e](https://github.com/CoreySpohn/eyepiece/commit/38a3a9e9d6d0828f11360b896b14db665e7dc209))

## [0.4.0](https://github.com/CoreySpohn/eyepiece/compare/v0.3.1...v0.4.0) (2026-08-28)


### Features

* **scene:** make the hidden-line convention trail's default depth cue ([85a7e75](https://github.com/CoreySpohn/eyepiece/commit/85a7e7513adc8edc3d8e156ed63db9e4bdf892aa))


### Bug Fixes

* **scene:** accept linestyle/lw under either alias in the hidden-line branch ([a93fa29](https://github.com/CoreySpohn/eyepiece/commit/a93fa29aa0c1f4afc90ed47239a8331928ce0caf))

## [0.3.1](https://github.com/CoreySpohn/eyepiece/compare/v0.3.0...v0.3.1) (2026-08-28)


### Bug Fixes

* **anim:** restore a no-engine figure instead of skipping the restore ([9540ba8](https://github.com/CoreySpohn/eyepiece/commit/9540ba826547001b82c0fb8169b48226a04d247d))
* **profiles:** take the IWA/OWA shading reach from the axes, not a constant ([7968230](https://github.com/CoreySpohn/eyepiece/commit/7968230fecabec67256025b78d7ee91b7589f6d5))

## [0.3.0](https://github.com/CoreySpohn/eyepiece/compare/v0.2.0...v0.3.0) (2026-08-24)


### Features

* add provenance stamps so a figure documents its own production ([ecc5c61](https://github.com/CoreySpohn/eyepiece/commit/ecc5c61d6a40cc812e67f2953d0f5100f693b07f))
* **anim:** warn when a data scale drifts mid-recording ([99516c0](https://github.com/CoreySpohn/eyepiece/commit/99516c04f8bd3aa80cfa2c10b9f7e339da5dd079))
* **provenance:** embed the stamp payload in file metadata ([4f356c6](https://github.com/CoreySpohn/eyepiece/commit/4f356c6b00ff43d3d9f0c2e4d405f4e31d58f8df))
* **stats:** let cov_ellipse name the interval it draws ([dade42d](https://github.com/CoreySpohn/eyepiece/commit/dade42d352a56c3ae22cf52dbd168d90d4959423))


### Bug Fixes

* **anim:** record through an Agg canvas so mp4 frames are not sheared ([8471aa1](https://github.com/CoreySpohn/eyepiece/commit/8471aa1a0ca394fb7e098f033fad59bd4ec4a0ff))
* **scene:** draw the sky_fan IWA disk under the tracks, not over them ([531d96a](https://github.com/CoreySpohn/eyepiece/commit/531d96a6dce76d701b8d40d7a1d98e6e30b9a437))
* **stats:** normalize the corner diagonal so an overlay lands on its scale ([15da2f0](https://github.com/CoreySpohn/eyepiece/commit/15da2f0a8b16c8f99dd3e85414444c06803e1b5a))

## [0.2.0](https://github.com/CoreySpohn/eyepiece/compare/v0.1.0...v0.2.0) (2026-08-21)


### Features

* **anim:** let save and video take an fps override like jshtml does ([bf69b48](https://github.com/CoreySpohn/eyepiece/commit/bf69b487b6fbb6fb7cabe5ad536e33291afbde03))
* **images:** add display_limits for data-derived vmin/vmax ([e158b85](https://github.com/CoreySpohn/eyepiece/commit/e158b85600ee6ec569a268a009a344a68b70ded5))
* **scene:** let trail take a SourceStyles entry so linked views compose ([ce5e462](https://github.com/CoreySpohn/eyepiece/commit/ce5e462b78bcfa502db176184f0cf17e1b776dc3))


### Bug Fixes

* **anim:** stop writer teardown from masking the real recording error ([7cafedb](https://github.com/CoreySpohn/eyepiece/commit/7cafedbb9abb26c937df7065086e23831f99447f))
* **images:** figure-level colorbar option and panel-count figure scaling ([01df23d](https://github.com/CoreySpohn/eyepiece/commit/01df23dc7fa2e9ed5bda11dbd740bd07a08bf666))
* **images:** hide index ticks without an extent, and let imshow_diverging update ([eab4e67](https://github.com/CoreySpohn/eyepiece/commit/eab4e676711405fc215cf53c600c9817507ddefa))
* **images:** size a triptych's owned figure for its three panels ([b3a175c](https://github.com/CoreySpohn/eyepiece/commit/b3a175c63bc4f5861700c2769e5b125ad7ca3b12))
* **profiles:** keep the IWA/OWA labels clear of the axes title ([2fbbe51](https://github.com/CoreySpohn/eyepiece/commit/2fbbe51a0d44c2bedd9861157748c638cfbdf746))
* **tests:** drop the mp4 sink from the zero-frame sink-naming test ([5b43320](https://github.com/CoreySpohn/eyepiece/commit/5b43320b16999101aea1ec7f2738fcc66fb2ae00))

## [0.1.0](https://github.com/CoreySpohn/eyepiece/compare/v0.0.1...v0.1.0) (2026-08-12)


### Features

* **anim:** export Animation publicly and freeze layout during record ([15cead4](https://github.com/CoreySpohn/eyepiece/commit/15cead4dbb296a534986da6db28eddea98370692))
* export ARTIST_KEYS and PRESETS from the top level ([3579750](https://github.com/CoreySpohn/eyepiece/commit/35797508e3c69df3daf6621b026f0a1c164c0994))
* **images:** add triptych and widen compare_row with vmin/vmax/imshow_kw/cbar_kw ([7fb443b](https://github.com/CoreySpohn/eyepiece/commit/7fb443bd7a7ecf279d3d2c6654abc4297826eb3c))
* **images:** forward extent/imshow_kw/cbar_kw through triptych and validate its inputs ([350579f](https://github.com/CoreySpohn/eyepiece/commit/350579f7933832c2634e36aceb6e0d6713141132))
* **profiles:** add plot_radial, plot_contrast_curve, radial_profile_plot ([e705346](https://github.com/CoreySpohn/eyepiece/commit/e7053465b222bdef0a54ad4cf80f56783ed5b83a))
* **schematic:** add rail(), the element-list optical-train primitive ([87946cf](https://github.com/CoreySpohn/eyepiece/commit/87946cfd0635849ab4491e432fd0388a09b1e3a6))
* **stats:** add axes= to corner for caller-supplied grids ([1406409](https://github.com/CoreySpohn/eyepiece/commit/14064091108454bb8c59a9d57ad719de1d7ef172))


### Bug Fixes

* **anim:** create missing parent directories for a recording sink ([fa9b02d](https://github.com/CoreySpohn/eyepiece/commit/fa9b02d0e116269424f54b4b810a754afba69e76))
* **anim:** skip the layout-engine restore for a figure that had none ([f60bdc8](https://github.com/CoreySpohn/eyepiece/commit/f60bdc8f36bb0cc161cf44c35d694866cfff438d))
* **docs:** correct the stale README status section and artist-key docs ([2b77449](https://github.com/CoreySpohn/eyepiece/commit/2b774496ec20e344e50da1261d0d925645208d62))
* **profiles:** anchor IWA/OWA shading to the axes edge and cycle colors per axes ([a2dbabc](https://github.com/CoreySpohn/eyepiece/commit/a2dbabc11194f1c4b200267e4a7636b58c5bf6f1))
* **profiles:** clarify contrast-curve floor and fill-key docstrings ([b2fe119](https://github.com/CoreySpohn/eyepiece/commit/b2fe119ca2c61c868e35f002eb698b860492c953))
* **profiles:** track contrast-curve annotations per marker kind and add axis labels ([ca3b51a](https://github.com/CoreySpohn/eyepiece/commit/ca3b51aa42ca60ed3eaeb83d7728ba91afda0b90))
* **schematic:** make the rail detector cap explicit and cover each glyph ([8c47f45](https://github.com/CoreySpohn/eyepiece/commit/8c47f45d59a108d7bd38402229f1ba5d11d75e55))
* **schematic:** raise ValueError for a non-string highlight ([b20eaaa](https://github.com/CoreySpohn/eyepiece/commit/b20eaaa96c9dc82600bb7934e145ce6f974e9324))
* **stats:** raise on title+axes and drop brittle golden-hash tests ([343e6e0](https://github.com/CoreySpohn/eyepiece/commit/343e6e07cb43d319350bc5ef0f928c738df5f830))
* **style:** fall back to light palette when the property cycle has no colors ([0341bdd](https://github.com/CoreySpohn/eyepiece/commit/0341bdd148e8fc2a4cdd76ff6faaedb11d2cec43))
* **style:** follow the user's prop_cycle in color() when no mode is active ([e4daebe](https://github.com/CoreySpohn/eyepiece/commit/e4daebe4fd850ee87acc10d984b799011fa72fcd))
* **style:** wrap the palette index and resolve neutral tones from rcParams ([bd432fb](https://github.com/CoreySpohn/eyepiece/commit/bd432fb0c760ecc648126979cc1eb0db3ff73b84))
* **tests:** use get_position(original=True) for geometry contract checks ([d162ffb](https://github.com/CoreySpohn/eyepiece/commit/d162ffb3d5054b21039bae911f6e31a6780d9668))

## 0.0.1 (2026-08-11)


### Features

* call-time style resolution with zero-style light fallback ([7888620](https://github.com/CoreySpohn/eyepiece/commit/7888620d77f6f5ac93379f8eab2d9733fea55376))
* complex-field 2x2 show_field with SubFigure embedding ([4c92480](https://github.com/CoreySpohn/eyepiece/commit/4c924806f2b9e790aa7d1b4a8facd68901482deb))
* corner plots, hist-vs-pdf overlay, covariance ellipse ([72da7be](https://github.com/CoreySpohn/eyepiece/commit/72da7be3d67a1560d7ef44f76bc54b6369b3a780))
* flat public namespace and firewall guard tests ([7d7b0a2](https://github.com/CoreySpohn/eyepiece/commit/7d7b0a208ce1113c9f8efaeec58bf05cc00a5fea))
* imshow_log, imshow_diverging, and shared-norm compare_row ([223c945](https://github.com/CoreySpohn/eyepiece/commit/223c9455007f2806c52919ce9ef7f6a750400447))
* mode-aware save_fig with directory anchoring ([74118de](https://github.com/CoreySpohn/eyepiece/commit/74118de9115036858dcb00308187a67a6ef9f87b))
* multi-sink grab-frame record() and animate() facade ([c217283](https://github.com/CoreySpohn/eyepiece/commit/c21728317626305f8c94bbdbe0401e26813c7962))
* pixel-edge extent and axis-label helpers ([47a052a](https://github.com/CoreySpohn/eyepiece/commit/47a052a56169595c569bee1ca0f386b4fd303a6e))
* PlotResult/MosaicResult return contract and artist key vocabulary ([cf3f06c](https://github.com/CoreySpohn/eyepiece/commit/cf3f06cac4273dda8181119b926fabb78429fba5))
* scaffold eyepiece package ([b08f5cb](https://github.com/CoreySpohn/eyepiece/commit/b08f5cb5ab550380df7d8e4232b0fcf584a94dbf))
* SourceStyles, Frame, and docs skeleton ([ea20c94](https://github.com/CoreySpohn/eyepiece/commit/ea20c947b4aa30a4064cd9ce1873c7480cb13e8b))
* trajectory trail with depth cues, sky_fan, fading_track, schematic rail ([5aa3967](https://github.com/CoreySpohn/eyepiece/commit/5aa39679828f726562a616f35fa1f9f17028469f))


### Bug Fixes

* compare_row single-image squeeze, empty-list guard, update range docs ([f6aae74](https://github.com/CoreySpohn/eyepiece/commit/f6aae741c5b147623833fdda462f101bdc904c36))
* identity-based eq/hash on PlotResult and MosaicResult ([ce629a3](https://github.com/CoreySpohn/eyepiece/commit/ce629a31dcdf66fd1ccba4175cb9a567f61cc4d1))
* restore internal-ref guard coverage with hook-safe pattern forms ([c693619](https://github.com/CoreySpohn/eyepiece/commit/c693619b35c042f1de4f615d62dd023ec850fa48))
* sky_fan errorbar artist, schematic lines key and highlight guard ([ba0cec3](https://github.com/CoreySpohn/eyepiece/commit/ba0cec34374f76c8ec0b142ae2b5536ad3f50e5c))
* validate configured ffmpeg path and cover anim mechanics with tests ([6c1613b](https://github.com/CoreySpohn/eyepiece/commit/6c1613b3d9ee664f6b6c4ec08a9f4388406dbc25))


### Miscellaneous Chores

* release 0.0.1 ([8b60e62](https://github.com/CoreySpohn/eyepiece/commit/8b60e62610407673d4f2426243831bb0784de8fd))
