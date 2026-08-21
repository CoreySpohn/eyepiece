# Changelog

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
