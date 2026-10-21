# Changelog

## Unreleased

- Add canonical task namespaces: `plot` for data displays, `clock` for
  scheduling, `motion` for moves, `style` for appearance, `render` for
  output, `testing` for checks, and `narration` for speech and captions.
  Existing root paths still work silently in 0.17. Deprecated aliases keep
  their existing warnings.
- Limit root star imports to `Timeline`, `Plan`, `Tempo`, `PlotResult`,
  `MosaicResult`, `MotionContract`, `animate`, `record`, `save_fig`, and
  `__version__`. Root discovery also lists the task namespaces.

## [0.16.0](https://github.com/CoreySpohn/eyepiece/compare/v0.15.0...v0.16.0) (2026-10-21)


### Features

* **anim:** pix_fmt encoder key, yuv444p opt-in with yuv420p kept as default ([485837e](https://github.com/CoreySpohn/eyepiece/commit/485837e544a32fc063b3a197c9c3e40a8316ff00))
* **anim:** reading PRESETS["talk_60"] warns, use "talk" ([591f06a](https://github.com/CoreySpohn/eyepiece/commit/591f06a25e5e098b4a301c3037f5fec4acb13dd5))
* **anim:** record draws frames unsnapped by default, add ep.unsnapped ([bd58cb4](https://github.com/CoreySpohn/eyepiece/commit/bd58cb49c4974f74bf8ca99ffcc5fbb06cebddb7))
* **anim:** remove record(allow_rescale=), deprecated in 0.15 ([1593d36](https://github.com/CoreySpohn/eyepiece/commit/1593d36c8fbb500a52db087441765b8925b35346))
* **anim:** talk preset records at 60 fps, docs presets stay at 30 ([4d41f2e](https://github.com/CoreySpohn/eyepiece/commit/4d41f2e3410fff12a41b6c2680904b81cc6c3ee4))
* **anim:** talk_rc sizes talk type and strokes from the frame height ([a1b9fb5](https://github.com/CoreySpohn/eyepiece/commit/a1b9fb515fbcef6fd8406e687136d96138281c37))
* **anim:** to_scrubber writes an offline HTML page whose slider scrubs one knob of a plan ([61d24f4](https://github.com/CoreySpohn/eyepiece/commit/61d24f42d116b0b6def01214ee2570e46aa9192f))
* **api:** add lazy aliases in task namespaces ([18e2200](https://github.com/CoreySpohn/eyepiece/commit/18e2200d2d7449631ec159331bc90edab74b679c))
* **api:** add plot, clock and motion namespaces and extend style, render, testing and narration ([604a2b6](https://github.com/CoreySpohn/eyepiece/commit/604a2b6644b405d8f053cd96b318730af4b34191))
* **api:** shrink the root to stable shortcuts, keep legacy root paths working ([6c6cc46](https://github.com/CoreySpohn/eyepiece/commit/6c6cc468c6604f785533b1bf176bb4590232da37))
* **attention:** add transient text underlines ([30a87b9](https://github.com/CoreySpohn/eyepiece/commit/30a87b97db23a15968fcf8157817595a06dbb29d))
* **attention:** circumscribe flash comet and a plateau pulse for indicate ([7322906](https://github.com/CoreySpohn/eyepiece/commit/73229060597361e8b77b67644aae8f6c011e5b59))
* **attention:** click lands a composite in one 0.6 s beat, then a 1.5 s hold ([ba8b907](https://github.com/CoreySpohn/eyepiece/commit/ba8b907592ebe81e650fc1737f6233cf036ba581))
* **attention:** head_draw behind= cuts halo and backing lines to the body so none leads the head ([0434e45](https://github.com/CoreySpohn/eyepiece/commit/0434e4533027914b2574caf4ef8c4c134391a824))
* **attention:** head_draw staggers by per-line offsets such as coordinates ([797b977](https://github.com/CoreySpohn/eyepiece/commit/797b9772d51cf35f9228dc05ced0e574072879e2))
* **attention:** head_draw, a constant arc-length reveal with a bright head ([405b3f2](https://github.com/CoreySpohn/eyepiece/commit/405b3f2fe98822f70ad69c22dfd1db195558679c))
* **attention:** remove the ignored write(overlap=) argument ([68c51e0](https://github.com/CoreySpohn/eyepiece/commit/68c51e0f6c58e010a1fcbaa060c30a827e2256c9))
* **attention:** ruler_readout binds a ruler and a counter to a drawing head ([8f09fed](https://github.com/CoreySpohn/eyepiece/commit/8f09fedcc3dc5f52023fa8ff4cf6dfb52683b340))
* **attention:** scaled arrow growth, staccato axis build, transform fade_toward ([3015963](https://github.com/CoreySpohn/eyepiece/commit/301596381dc70f4b2f3359e1ed6f728b928ab93f))
* **attention:** Stage.weigh lift_pt lands a label into place with its fade ([097d28a](https://github.com/CoreySpohn/eyepiece/commit/097d28a2e367373f676eef4a187bb30408c51d45))
* **attention:** write traces each glyph with a thin pen, then fills it ([7cfb69f](https://github.com/CoreySpohn/eyepiece/commit/7cfb69f7f4f167e941c314683ce77ea9e1b3cec0))
* **clock:** declare zero duration cuts ([9a34389](https://github.com/CoreySpohn/eyepiece/commit/9a34389d3a0743b9eb20b62d59e616c0c9e545b5))
* **clock:** Plan.spans_json(holds=False) leaves holds out of a manifest ([72b2c78](https://github.com/CoreySpohn/eyepiece/commit/72b2c78e91e992dc3febb30389b927fcddc3bf6e))
* **clock:** rename Timeline.pulse and ring to indicate and circumscribe ([f3eeb7b](https://github.com/CoreySpohn/eyepiece/commit/f3eeb7bb93a68f746b8b86430581d61f4737d4f0))
* **clock:** text naming a move may arrive over its last quarter ([874b8ec](https://github.com/CoreySpohn/eyepiece/commit/874b8ecf574268ca1c7d2e81eb9803fe9e8112a4))
* **clock:** Timeline.spans, a read-only live view of the declared spans ([3d76e5a](https://github.com/CoreySpohn/eyepiece/commit/3d76e5ac0a51819ecbe92cb1bf5c31c0c95536f2))
* **clock:** Timeline.wait_until holds the picture until a time without a span ([0f9c0c2](https://github.com/CoreySpohn/eyepiece/commit/0f9c0c2361f69971bd1c7ee4504a906a4eba0890))
* **equation:** bind a term to the mark it names, with linked cues ([1a8096f](https://github.com/CoreySpohn/eyepiece/commit/1a8096f08f3d6212f42212b97586901aa54b553f))
* **hop:** exchangeable draws with dimming ghosts and a converging tally ([51a442a](https://github.com/CoreySpohn/eyepiece/commit/51a442acee3ff4dfef64e7ee52063707931cbd47))
* **hop:** warn at a rate outside the three to six draws a second band ([f4311a4](https://github.com/CoreySpohn/eyepiece/commit/f4311a40c69ddfce76649ecda512123db6050fae))
* **labels:** spread nearby curve end labels ([4df8821](https://github.com/CoreySpohn/eyepiece/commit/4df88212af708076374acfec3137ab24c81b4f3c))
* **marks:** rename spans to shade_intervals, spans deprecated ([4f7c6ec](https://github.com/CoreySpohn/eyepiece/commit/4f7c6ec00725f3175294ffaa1c61fdfda5065b85))
* **motion:** add the smooth sigmoid easing and a DEFAULT_EASE constant ([3cb3a54](https://github.com/CoreySpohn/eyepiece/commit/3cb3a546119b18d90a9998f584f275f2dbf939a1))
* **motion:** default easing is the smooth sigmoid, read from DEFAULT_EASE ([c02695d](https://github.com/CoreySpohn/eyepiece/commit/c02695d70269e80bdcc0e4c6fcc3c0a2d4411401))
* **motion:** expose the plateau pulse shape ([d5cdbb7](https://github.com/CoreySpohn/eyepiece/commit/d5cdbb755f6f15e976af679cc6497853288796ba))
* **narration:** a 20 s sweep prototype with its count heard, cued and silent ([e375264](https://github.com/CoreySpohn/eyepiece/commit/e3752647003b2ed1a8bbd835b1082aa5c3196279))
* **narration:** audio_track mixes cue sounds before the loudness pass; the voice yields to them ([3c4909e](https://github.com/CoreySpohn/eyepiece/commit/3c4909e5baf3ea244876fe8763f6a05e9c4f0905))
* **narration:** check_number_sync, a readout settles on its spoken number ([8ef5d7c](https://github.com/CoreySpohn/eyepiece/commit/8ef5d7c1d10a7cd2e1b3f120d7dc22679cd0d879))
* **narration:** cue synthesizers for a count tone, pixel-read ticks and a click sting ([9f40b96](https://github.com/CoreySpohn/eyepiece/commit/9f40b96b1cd6abed1d9be0c6e84964eb249c4820))
* **narration:** duck windows drop the room-tone bed to its floor ([7e6aecf](https://github.com/CoreySpohn/eyepiece/commit/7e6aecf7ba3e39faa770075869e1342a552a7e2d))
* **narration:** Line sentence_gap_s speaks a line one sentence at a time ([f116ca4](https://github.com/CoreySpohn/eyepiece/commit/f116ca4f99a7fcf8090d3b31a266a143a78ed2bc))
* **narration:** lint_script and check_script_speech rules for sentences, numbers and cues ([195dfdd](https://github.com/CoreySpohn/eyepiece/commit/195dfdddf72f40f17159fb0f26844fee941bf4d8))
* **narration:** pair scientific screen and speech values ([b80dc93](https://github.com/CoreySpohn/eyepiece/commit/b80dc932175098aa4d72c1fff152e3051c95f661))
* **narration:** Pause, a first-class silent beat with a duration and an on-screen description ([64096f6](https://github.com/CoreySpohn/eyepiece/commit/64096f61662ee96e298ab1e8778d4d3884a57f81))
* **narration:** per-line speech speed that enters the line's cache key ([7aab0e9](https://github.com/CoreySpohn/eyepiece/commit/7aab0e97c56c138a1e67c992dfc823bcfece8a79))
* **narration:** remove Narration(min_s=), deprecated in 0.15 ([d9451bf](https://github.com/CoreySpohn/eyepiece/commit/d9451bfce30230a0ebadfca187bfb7d029bfefbd))
* **narration:** silence lint for cue coverage, undeclared silence and the hero beat ([f00034b](https://github.com/CoreySpohn/eyepiece/commit/f00034b56ff9809be87925bb8eecf6eac8fc0412))
* **narration:** synthesize takes rules= to refuse lint findings ([190af7e](https://github.com/CoreySpohn/eyepiece/commit/190af7e40988412a3b272b6bded90807339d0675))
* **narration:** write_chapters, a WebVTT chapters file with one cue per beat ([a0550e5](https://github.com/CoreySpohn/eyepiece/commit/a0550e5d0f7551d53ef25a6fc0004a4c8009de32))
* **photons:** aperture_counts reads counts and known-leak S/N off the photons mark ([b08eb5b](https://github.com/CoreySpohn/eyepiece/commit/b08eb5b31877be3a052852d051f89cec05fae060))
* **photons:** photons mark with settling arrival glow and exposure_text log clock ([ed678e7](https://github.com/CoreySpohn/eyepiece/commit/ed678e7ed81a7da7efdca181d2f1e25c60a227ce))
* **photons:** seeded Poisson arrivals on an exposure clock, final frame fixed by the seed alone ([0a37e13](https://github.com/CoreySpohn/eyepiece/commit/0a37e139ab23763dde44db3c499cc806ffcf2799))
* **profiles:** name image profile computation explicitly ([53ae2d3](https://github.com/CoreySpohn/eyepiece/commit/53ae2d3ae0a3d6fdc441bd62837b147bc8065437))
* **relay:** add carried panel furniture tags ([88a4c31](https://github.com/CoreySpohn/eyepiece/commit/88a4c3189704afe504edb7d2b32973abf99b0cb0))
* **render:** content keys over a clip's inputs ([93d01e0](https://github.com/CoreySpohn/eyepiece/commit/93d01e0d451b43883ad78763bafba3b37f9667bb))
* **render:** deterministic manifest of rendered variants ([6d1e4a0](https://github.com/CoreySpohn/eyepiece/commit/6d1e4a0af4a0e3a76dfd5da295f815a2c6249e12))
* **render:** deterministic posters, stills and picture comparison ([bb3e212](https://github.com/CoreySpohn/eyepiece/commit/bb3e2126f8ebc38b9fd5ccd97898efe9580b20f6))
* **render:** let clip specs choose variant output paths ([1067126](https://github.com/CoreySpohn/eyepiece/commit/1067126df14f9b0fb7d192b5f93db1f256e83e41))
* **render:** per-variant frames and adoption of videos recorded before keys ([0a35507](https://github.com/CoreySpohn/eyepiece/commit/0a355074aa3c3e612509855288548cdf4ee35043))
* **render:** render frames in worker subprocesses with resumable chunks ([310aae3](https://github.com/CoreySpohn/eyepiece/commit/310aae385c8d10d40f87d067a83a5e452f9eecf0))
* **render:** scale-drift check for parallel renders ([daa205c](https://github.com/CoreySpohn/eyepiece/commit/daa205cd8bb0cf0d672bafc7580d973896a8af5c))
* **render:** split the driver into private modules behind the render facade ([f5f92d9](https://github.com/CoreySpohn/eyepiece/commit/f5f92d942c47afb4a02fb42ab3662a1c6091c6c9))
* **render:** venue, mode and narrated variants keyed on inputs ([d48f972](https://github.com/CoreySpohn/eyepiece/commit/d48f972eda7c01d9609ad436762bb7cafbf54578))
* **style:** add quiet axes spines ([499ad6f](https://github.com/CoreySpohn/eyepiece/commit/499ad6f3fd1b87bad2e24669bafd170309db9881))
* **style:** label halo drawn at 0.6 opacity ([44475ed](https://github.com/CoreySpohn/eyepiece/commit/44475edd340449e854a1e1f07214cc4199d9264d))
* **style:** role takes neutral ranks and mode=, resolve_role deprecated ([b027d5e](https://github.com/CoreySpohn/eyepiece/commit/b027d5eb188a5af2f1fcdd1e3a7454f016c37b02))
* **style:** thinner label halo, denser dark region fill, butt caps in VIDEO_RC ([03fb7da](https://github.com/CoreySpohn/eyepiece/commit/03fb7da6ba42381382bb2bfc29573c74e84d60a4))
* **style:** video-safe dark roles derived from dark, gated on motif_faults ([0c49312](https://github.com/CoreySpohn/eyepiece/commit/0c493128f18ddb1049e0601b6eb5fc9850fa48dd))
* **testing:** draw groups excuse only their own cuts at declared steps ([5d51a2b](https://github.com/CoreySpohn/eyepiece/commit/5d51a2b43b09efed0a54b7e5f04f5871408c33b4))
* **testing:** grounding_faults finds symbols with no referent, the wrong color, or ahead of their mark ([ad537d2](https://github.com/CoreySpohn/eyepiece/commit/ad537d23c9738ce766f8257c59e78b1a7c5d57f9))
* **testing:** hop groups of exchangeable draws in motion_faults and check_clip ([35ce556](https://github.com/CoreySpohn/eyepiece/commit/35ce556bdc02067e26e1e57ec5c55a159b711a62))
* **testing:** lint text halos against the backing-box convention ([12e64eb](https://github.com/CoreySpohn/eyepiece/commit/12e64eb167bfa38cfb69f7d52c0d621611fe317d))
* **testing:** motion_ink ratio of driven pixels in motion_ink and check_clip ([da3d29e](https://github.com/CoreySpohn/eyepiece/commit/da3d29e78bf00a5112c871ea1ed2ac8385c7b15b))
* **testing:** rename the check window Span to Window, Span deprecated ([89b30c4](https://github.com/CoreySpohn/eyepiece/commit/89b30c47a55422bac0fde51fd9c83a54b6cd65ab))
* **testing:** review_sheet writes one image of a clip at its event times, with captions and waveform ([3c323f5](https://github.com/CoreySpohn/eyepiece/commit/3c323f5e1cfcf82065d440799edae8f3739613d9))
* **testing:** timing_faults checks speech rate, line gaps and silence in one call ([4af1815](https://github.com/CoreySpohn/eyepiece/commit/4af181532a35da47833a0da9054d6423e1060328))
* **testing:** timing_faults finds a number that lands after its word or leaves it while said ([c38e9cc](https://github.com/CoreySpohn/eyepiece/commit/c38e9cce265062c1501237b3f02d191db01c1109))
* **testing:** timing_faults finds a still screen under more than 2 s of speech ([fedc14b](https://github.com/CoreySpohn/eyepiece/commit/fedc14b1484f8d37c28b54ecc47133c4e9c05b3f))
* **weighted_draws:** weights drawn as area or opacity with a resampling beat ([2abac33](https://github.com/CoreySpohn/eyepiece/commit/2abac333d17d52723baa75aa85c63251e0ffa533))


### Bug Fixes

* **alpha:** keep mesh opacity changes local to the target ([2031575](https://github.com/CoreySpohn/eyepiece/commit/203157538384b176fee9cedec4a45b8fae16d456))
* **alpha:** preserve mesh face opacity through fade and Stage ([23f2ae4](https://github.com/CoreySpohn/eyepiece/commit/23f2ae4ea1e9f26fa5965a6387619595bbd26b17))
* **anim:** scrubber slider is labelled with the knob and its unit, not the page title ([596b6ce](https://github.com/CoreySpohn/eyepiece/commit/596b6ce8dc0d39c1f8110e98bd0512726160fde5))
* **attention:** accept write(overlap=) with a deprecation warning; clarify write, ruler_readout and mode docstrings ([dfbcfb8](https://github.com/CoreySpohn/eyepiece/commit/dfbcfb88abb42983f0ff7345efcb984e51c4ec6a))
* **attention:** build_axes resolves titles at update time ([5ae52e1](https://github.com/CoreySpohn/eyepiece/commit/5ae52e1ca2dcfc8a8a69dd00e21d6461e90d1119))
* **attention:** preserve underline color and text opacity ([487aa73](https://github.com/CoreySpohn/eyepiece/commit/487aa734cadcdd3d3bb565a3a8658a8b025ac4eb))
* **brush:** name veil opacity explicitly ([67ad752](https://github.com/CoreySpohn/eyepiece/commit/67ad752dab1d1e091e5c4807999f5a598142f8da))
* **clock:** a click placed with at keeps its hold right after the beat ([506626d](https://github.com/CoreySpohn/eyepiece/commit/506626df449de79741c9a91c8831776db8e90cf0))
* **clock:** clearer error when names= refers to an unknown span ([4877713](https://github.com/CoreySpohn/eyepiece/commit/4877713ed5157255c9b975388ce927f3d19225e6))
* **deps:** add ipython to the test extra for the gallery grounding test ([02095fa](https://github.com/CoreySpohn/eyepiece/commit/02095fa3598ffb7c51fa304b76046212d10d9db7))
* **deps:** require hwostyle 1.7.0 for the accepted motif pairs ([5064a27](https://github.com/CoreySpohn/eyepiece/commit/5064a27d7fde0d244cffcfe0494978b5ad2829ba))
* **docs:** apply timeline reveal easing once ([1a345ef](https://github.com/CoreySpohn/eyepiece/commit/1a345ef5b7e48213c98f5b9d896e47de6353cd31))
* **docs:** keep the measurement head inside the axes ([9f98f0e](https://github.com/CoreySpohn/eyepiece/commit/9f98f0e1249d5d7eb5f19a226b69fafd26c043d5))
* **gallery:** stack the orbit counter, key and sky from measured text extents ([d5933cd](https://github.com/CoreySpohn/eyepiece/commit/d5933cd2faf8387cc5b6820e9bdf938216ea5b91))
* **hop:** drop internal vocabulary from the docstring and guard against it ([71a7f44](https://github.com/CoreySpohn/eyepiece/commit/71a7f4446c69388e681446a29e2cda99a47c2741))
* **images:** ruler label gap is measured to its backing box edge ([a4d8772](https://github.com/CoreySpohn/eyepiece/commit/a4d8772f3e4e9a0d06fb20b486ee86f6a05ff927))
* **motifs:** add the dark model-negative record to the stand-in accepted pairs ([10c19c7](https://github.com/CoreySpohn/eyepiece/commit/10c19c742da08774affb2be675b08e150aed5e08))
* **motifs:** mirror the paper zodi-negative and prior-positive accepted pairs ([bf24c64](https://github.com/CoreySpohn/eyepiece/commit/bf24c6406b459f3ea9b7c54849d05774ff6b7f25))
* **motion:** compare hop crops before building cut masks ([5b46d0e](https://github.com/CoreySpohn/eyepiece/commit/5b46d0e5976de8c4324ae9ef5cf37a1ebd4144b7))
* **motion:** count hops before adding caption masks ([edb4e29](https://github.com/CoreySpohn/eyepiece/commit/edb4e29d6b182ccc4c963417e775daac64bf60e7))
* **motion:** include a cut move's declared captions ([1362649](https://github.com/CoreySpohn/eyepiece/commit/1362649d01fa315ae5a310856a9e3138a8d73659))
* **motion:** name artists that never become visible ([0e6fe1e](https://github.com/CoreySpohn/eyepiece/commit/0e6fe1ebbcc43ef4dc3fcecb3b0253bbbe2ed0be))
* **motion:** report every excessive hop window ([7b12711](https://github.com/CoreySpohn/eyepiece/commit/7b12711682a345ac2b784ceced80e906ef932ac8))
* **motion:** tolerate one-level fade quantization ([b2544e4](https://github.com/CoreySpohn/eyepiece/commit/b2544e407f0ecec3587eae88b598d31c5e9e6202))
* **narration:** bound spoken scientific exponents ([7ec9190](https://github.com/CoreySpohn/eyepiece/commit/7ec9190b32f83a06a3cc6ff56a2cd529269f9c81))
* **narration:** chapters docstring matches the cue end; refuse an empty title ([6a12049](https://github.com/CoreySpohn/eyepiece/commit/6a1204945a947625c111d2fe4832f15714054c4c))
* **narration:** count a bare one as a number only beside a quantity word ([a01106a](https://github.com/CoreySpohn/eyepiece/commit/a01106a4a1504636ee7f414cf743d5125b670b48))
* **narration:** match muxed audio to video presentation duration ([9e56dc3](https://github.com/CoreySpohn/eyepiece/commit/9e56dc306c5402d14cebe20b570e8fee6d9aff75))
* **narration:** measure a late counter over its span, flag a departure from the spoken value, floor the settle at one frame ([a5566f4](https://github.com/CoreySpohn/eyepiece/commit/a5566f4a27d2339b1e14b200ca1e99a422ea6c0c))
* **narration:** preserve sting decay at the clip boundary ([7876fc1](https://github.com/CoreySpohn/eyepiece/commit/7876fc137c68b11daa9de12ec896b24ece5f6c78))
* **narration:** restore sweep style after rendering ([c56a43a](https://github.com/CoreySpohn/eyepiece/commit/c56a43a99dce90f3e5d8b307ebc24a3a5507ccfd))
* **narration:** reuse the WAV writer in the sweep example ([b581979](https://github.com/CoreySpohn/eyepiece/commit/b581979785e806ac18f8823cb6b2293209b8f94b))
* **narration:** stop flagging unambiguous slashes and hyphens; synth --estimate refuses cache flags ([8e95b8a](https://github.com/CoreySpohn/eyepiece/commit/8e95b8a97ba92b7be2eac831f7412cc90215ab66))
* **render:** isolate narration scratch files ([3c968cf](https://github.com/CoreySpohn/eyepiece/commit/3c968cf01991219ea8cb73c6ddcdd0a8427efb55))
* **render:** keep version metadata at the package root ([905e737](https://github.com/CoreySpohn/eyepiece/commit/905e73724fec4441b92ca4784e6fabf4667a8180))
* **render:** key package source and clean successful workspaces ([8d3a139](https://github.com/CoreySpohn/eyepiece/commit/8d3a139b2fba757b2b7ca077d2f3354cf2a7c7d2))
* **render:** match record raster sizing and draw state ([6a47074](https://github.com/CoreySpohn/eyepiece/commit/6a47074e7be03c5d08a54a2eb6207adb3265fdbd))
* **render:** pass still savefig overrides through ([39487cd](https://github.com/CoreySpohn/eyepiece/commit/39487cd69e009ce39ff839b402ac7cf1743fc736))
* **style:** make blend canonical while retaining motion alias ([4db53ef](https://github.com/CoreySpohn/eyepiece/commit/4db53efc573c29c35e6120b61578b2d6022705dd))
* **testing:** count a backing box at its face-color alpha alone ([066f097](https://github.com/CoreySpohn/eyepiece/commit/066f0974df06c906a3e62b2cc175f22d227b29c9))
* **testing:** detect halos on text with transparent fill ([a58694c](https://github.com/CoreySpohn/eyepiece/commit/a58694cb4c32edd650fd28160885ca6eb2432fe4))
* **testing:** make Fault canonical while retaining clock alias ([8219353](https://github.com/CoreySpohn/eyepiece/commit/8219353aae3f45f72109bd975fd2d648d70a5c4a))
* **testing:** match scientific speech with relative tolerance ([0bdcc5b](https://github.com/CoreySpohn/eyepiece/commit/0bdcc5bd80533816ba8f7e6af9197fa75edee968))
* **tests:** keep WAV coverage active without video tools ([37df04a](https://github.com/CoreySpohn/eyepiece/commit/37df04a82d5cd49198b8a594f91706c99f1fd443))
* **timing:** count cue-linked arrivals on still screens ([2fd57c0](https://github.com/CoreySpohn/eyepiece/commit/2fd57c039323f3a970b52824607d1cc6bbe71532))


### Performance Improvements

* **testing:** index motion overlap allowances by artist pair ([2c1d268](https://github.com/CoreySpohn/eyepiece/commit/2c1d268e6cc9db9183e8914b142ee30469a49ee0))

## [0.15.0](https://github.com/CoreySpohn/eyepiece/compare/v0.14.0...v0.15.0) (2026-10-16)


### Features

* **anim:** add shared encoder profile, 30 fps presets, unsnapped moves and seamless phase ring ([29a8aa5](https://github.com/CoreySpohn/eyepiece/commit/29a8aa5924ef39893c233ee608bb7d08f3015172))
* **anim:** record stills= during the encode and on_rescale= replacing allow_rescale ([868b511](https://github.com/CoreySpohn/eyepiece/commit/868b511df1077f8a982d9fabb111a5fee7f7d398))
* **anim:** record warns once about unread and constant knobs, with end-to-end stills tests ([730f527](https://github.com/CoreySpohn/eyepiece/commit/730f5278467dcdaf66cd22f90ae616c45fc8acac))
* **arrows:** add arrow, angle_arc and compass, marks that grow, sweep and turn on one artist ([c6809c5](https://github.com/CoreySpohn/eyepiece/commit/c6809c5319d3c1221530aabbd9d1a317ddfefb6d))
* **attention:** add follow_zoom, a magnified inset that follows a moving point ([a388ae7](https://github.com/CoreySpohn/eyepiece/commit/a388ae7b964203db9bf9befdad7d67a5a5d8b829))
* **attention:** add morph, a ghost that carries one path into another ([3c515a8](https://github.com/CoreySpohn/eyepiece/commit/3c515a8dcfbb90ef71990e9bfc0e85f93a551071))
* **attention:** add readout, a number that counts in a fixed layout ([63e0541](https://github.com/CoreySpohn/eyepiece/commit/63e0541c4decb871dd558b505e14bcf7a4a1435a))
* **attention:** add words, one line of text with per-word color and in-place reveal ([9e5b355](https://github.com/CoreySpohn/eyepiece/commit/9e5b355470d310d8d254d34acd69e514d92a0c49))
* **attention:** add write, a handwriting reveal that traces glyph outlines then fills them ([025c3ac](https://github.com/CoreySpohn/eyepiece/commit/025c3ac48cba5dd5e557e1d923b52274effe6c3e))
* **attention:** draw_on reveals a line along its display-space length ([cecf0e4](https://github.com/CoreySpohn/eyepiece/commit/cecf0e4da5c4eadf3639b76faec7724077c7124c))
* **attention:** ease the presence of a stage group in weigh ([bc58cab](https://github.com/CoreySpohn/eyepiece/commit/bc58cab244a94dbd0184a4aa5a7b0e1173d1fa31))
* **attention:** ep.warp and ep.warp_grid, matrix and pointwise plane warps ([c80f282](https://github.com/CoreySpohn/eyepiece/commit/c80f28280c2919c303d5704fed1527e14c2db260))
* **attention:** follow_zoom halo for dark images and loc auto with hysteresis ([f7f853e](https://github.com/CoreySpohn/eyepiece/commit/f7f853efc611c8a3df293af11e7306feb77a3fe1))
* **attention:** let words colors take a word string as a key ([edb4216](https://github.com/CoreySpohn/eyepiece/commit/edb4216a8b29e1a675d7108a7eb15282b94b51f3))
* **clock:** one timeline of named spans, knobs derived at build time, and the events of the plan ([5bac655](https://github.com/CoreySpohn/eyepiece/commit/5bac65575522bc5ae6f0d05d54a154d8b65cf57f))
* **clock:** Plan.value reads the t, index and section frame fields ([d67fd73](https://github.com/CoreySpohn/eyepiece/commit/d67fd73c19862d46c97b1f5164590a604a7523aa))
* **equation:** add equation and transform, glyph-level math with named terms and a move between two ([5bb92a7](https://github.com/CoreySpohn/eyepiece/commit/5bb92a7ac8b32737e263972bbd28b4a5af18da4c))
* **equation:** add replace=morph to transform, outline-to-outline as Manim aligns paths ([2996833](https://github.com/CoreySpohn/eyepiece/commit/2996833e82f3d1ab8f6481390e66f83395a82dfe))
* **equation:** align, group motion, replaced terms and travel-scaled arcs in transform ([1a635d7](https://github.com/CoreySpohn/eyepiece/commit/1a635d75f700a5dde270e7f5a98e94dc9977b924))
* **equation:** arc by travel direction and align on the shift most groups share ([17e5927](https://github.com/CoreySpohn/eyepiece/commit/17e592732ceb80cd5b57d8d5eb2b252fad7f3b73))
* **equation:** make morph the default for replaced terms ([e573511](https://github.com/CoreySpohn/eyepiece/commit/e573511a07a8626feda7c5b2048d788d780a8285))
* **motifs:** add Motif and MotifSet with motif_faults checks ([16f43fd](https://github.com/CoreySpohn/eyepiece/commit/16f43fd0a8e6df26190611da992aa70eeae0518e))
* **motifs:** motif kwargs carry a gid and MotifSet.meaning_keys ([d9b1d34](https://github.com/CoreySpohn/eyepiece/commit/d9b1d34b049f646b724a9b714fb1c65a94a1a17d))
* **motifs:** read label inks and accept declared pairs with a second channel ([94f0b26](https://github.com/CoreySpohn/eyepiece/commit/94f0b26d36f525a125089677adf70f13577b38d6))
* **motion:** add back_out and running_start eases with a strength ([afd794a](https://github.com/CoreySpohn/eyepiece/commit/afd794afe3ca43ac4e378a0358a05befc4c4b8f9))
* **motion:** MotionContract on move results, declared for the gallery moves ([1a14886](https://github.com/CoreySpohn/eyepiece/commit/1a14886668e8170c52b4c37f1dd37acf05a217b7))
* **narration:** caption_at, and caption breaks avoid brackets and bare operators ([a6bdd9f](https://github.com/CoreySpohn/eyepiece/commit/a6bdd9fe9dfc8e3d154c69aca9fcc1c41fe04b34))
* **narration:** JSON scripts, --estimate, and a caption sentence-count warning ([ece0e2b](https://github.com/CoreySpohn/eyepiece/commit/ece0e2b24572f1cd0d1f941ab6cc6e89a92d9c95))
* **narration:** keep min_s= as a deprecated duration from the line start ([53354b9](https://github.com/CoreySpohn/eyepiece/commit/53354b9600fb2a8c77c441b144147243cc748811))
* **narration:** loudness, room tone, wrapped captions, caption placement and display text ([28a4985](https://github.com/CoreySpohn/eyepiece/commit/28a4985c5fb4a736c68d46af3b657986a3aecd9d))
* **narration:** say and speech_faults for numbers the voice misreads ([1981462](https://github.com/CoreySpohn/eyepiece/commit/1981462603b591a8e4f9a855eb1e93a0e4f86f20))
* **narration:** speak a clip script and cue attention moves on its words ([7683422](https://github.com/CoreySpohn/eyepiece/commit/76834229e33746238d9d60963f332aa4f71aba17))
* **output:** deterministic save_fig and stamp(reserve=True) ([e52d52d](https://github.com/CoreySpohn/eyepiece/commit/e52d52d6ace319b8dd9bd9879d91ae0a620df3de))
* **style:** pupil colormap role, corner density on probability ([eeded1d](https://github.com/CoreySpohn/eyepiece/commit/eeded1d74e0c597197ba78be7b03dcede9504efe))
* **testing:** add meaning_faults color-meaning check ([6ea8e5b](https://github.com/CoreySpohn/eyepiece/commit/6ea8e5bb6f9fe00eed67cbf4d14812f38cc08b19))
* **testing:** add motion_faults and assert_motion, a linter for the path between states ([b6c4676](https://github.com/CoreySpohn/eyepiece/commit/b6c4676fc0459e7bcdce9e6dd86f9ef3ea21cc39))
* **testing:** add speech_rate and assert_speech_rate ([fb93e2e](https://github.com/CoreySpohn/eyepiece/commit/fb93e2e320e6e201c98a201065c58c12dca5066b))
* **testing:** check_clip and assert_clip check a plan-driven clip in one call ([7898f24](https://github.com/CoreySpohn/eyepiece/commit/7898f24b7d319ade2542f99940df628db14dec49))
* **testing:** declared steps for the motion check ([82ba1a4](https://github.com/CoreySpohn/eyepiece/commit/82ba1a41228928227fccc21ee092fbdfefed6a19))
* **testing:** readability checks for a figure or a clip sampled at times ([98dbd89](https://github.com/CoreySpohn/eyepiece/commit/98dbd895f3571e061978953f3cd6670eb9bb5d71))
* **testing:** shape-class fault kind for pieces that must stay axis-aligned rectangles ([f8e8da2](https://github.com/CoreySpohn/eyepiece/commit/f8e8da26b6e13773f33f454193884e30f71ebb50))
* **video:** add the video extra and name it in the missing-ffmpeg error ([fd3ccaa](https://github.com/CoreySpohn/eyepiece/commit/fd3ccaaa2ad53884bca03206f07e81d865d7479f))


### Bug Fixes

* **anim:** fade in the real text during write so the end has no jump ([d29c02c](https://github.com/CoreySpohn/eyepiece/commit/d29c02cd4e768751e8b23841185836166d3e094e))
* **attention:** draw_on rejects stepped drawstyles ([c771d88](https://github.com/CoreySpohn/eyepiece/commit/c771d88737fa82d8357e535746295ec90f9f599d))
* **attention:** readout ink test, clamp overshoot eases, reject negative when signed=False, term_row wording ([3714b7b](https://github.com/CoreySpohn/eyepiece/commit/3714b7b3690733411dcd6a6e73d0c36ba3478251))
* **clock:** max_late_s, narrated captions skip short-read, separate label namespace, truncating clamp, chaining stagger ([099010d](https://github.com/CoreySpohn/eyepiece/commit/099010db1113f084ac5ec943420a9fbdbd3c9b13))
* **equation:** draw curved glyphs as cubics, morph contract overlap, morph gallery prose ([e8043dc](https://github.com/CoreySpohn/eyepiece/commit/e8043dcab48f877ef9fd3d59bc581898371bb4c7))
* **equation:** land on the snapped rest state, so the ends of a transform do not jump ([9575b81](https://github.com/CoreySpohn/eyepiece/commit/9575b81ed5edb75200ba5ec3cf306948652ddcc0))
* **equation:** pair tagged with untagged equations by glyph order and warn on a lost tag partner ([8341755](https://github.com/CoreySpohn/eyepiece/commit/8341755c84ec3a919c14de9cbf804f337d04254e))
* **equation:** stretch a fraction bar as a rectangle, so its ends stay square ([49d0a84](https://github.com/CoreySpohn/eyepiece/commit/49d0a84cea3839e82e544702152d403263902e94))
* **motifs:** check roles in sharing, full stand-in table, mutation tests, docs extra ([cf8b4ce](https://github.com/CoreySpohn/eyepiece/commit/cf8b4cea8e2e3b52ce5f3525ded6bcdcb194cf79))
* **motion:** continuous collapse copy, morph ghost and follow_zoom halos; gallery label room and unroll ([f05d671](https://github.com/CoreySpohn/eyepiece/commit/f05d671e8de5c35de2a59c6be12dfdcc2258bc70))
* **motion:** write declares nothing, transform rectilinear skips replaced terms, negative controls for the contracts ([1b65f3c](https://github.com/CoreySpohn/eyepiece/commit/1b65f3cba84c320b44fbb5a29035dc7a8c7c4e75))
* **narration:** add the room-tone bed after loudness normalization ([979c42b](https://github.com/CoreySpohn/eyepiece/commit/979c42b1c62502d7ed82f600cfca1b3dbe801452))
* **narration:** name the pinned kokoro version in the missing-soundfile error ([d32f666](https://github.com/CoreySpohn/eyepiece/commit/d32f6660620e9ac98986899ed19ed98f10dcb899))
* **testing:** check figures whose canvas cannot return pixels ([072cf2c](https://github.com/CoreySpohn/eyepiece/commit/072cf2c25748aad199b25fa778fe07b16a564475))
* **testing:** check_clip ignores info-level plan findings ([f9e4881](https://github.com/CoreySpohn/eyepiece/commit/f9e4881fa199ecf96987e023ffd9d77024dc6951))
* **testing:** check_clip takes NumPy integer step frames and refuses bools ([af72d25](https://github.com/CoreySpohn/eyepiece/commit/af72d25023edfac07405e52f942fbef96dfc5ab0))
* **testing:** check_clip validates declared steps ([a39a961](https://github.com/CoreySpohn/eyepiece/commit/a39a9610e6943bdd190057a9b83110f7e46d9ff7))
* **testing:** honor may_overlap over the motion window at a span's ends ([b9ea62f](https://github.com/CoreySpohn/eyepiece/commit/b9ea62f6b8695bc6048ea97a05295cfbf6a40381))
* **testing:** ignore reveal masks, tick labels and faint overlaps in motion_faults ([aa86b8d](https://github.com/CoreySpohn/eyepiece/commit/aa86b8d99452408a65f89da2fb39bd147a5021b7))
* **testing:** lone-step teleport rule only where the pixel baseline is empty ([eab2e70](https://github.com/CoreySpohn/eyepiece/commit/eab2e70cfc9fe815c1e9e9029f449a23193243ad))
* **testing:** meaning check faults on shared pairs, cue hue, dimmed advisory, rebind in times ([522edc6](https://github.com/CoreySpohn/eyepiece/commit/522edc6e8370b29ffc4d966a8e3f28ac2b1fc3bc))
* **testing:** motion_faults review fixes (contract dicts, collections, teleports, overlap, rest, spikes, cost) ([730bf28](https://github.com/CoreySpohn/eyepiece/commit/730bf280141a1dd6ecf942b713bce3dcf49bc89d))
* **testing:** motion_faults second review (collection pairing, lone teleports, reveal clip removal, rest boundaries, spans, images) ([bbee369](https://github.com/CoreySpohn/eyepiece/commit/bbee369a1bc877732d151096cac17f67ea295b4f))
* **testing:** shape-class treats a straight line as a thin rectangle and keeps hairline corners ([3516f6f](https://github.com/CoreySpohn/eyepiece/commit/3516f6f7798f5a1688463699a9f688a4bee7851f))
* **warp:** level no longer scales the warp, so dimming never undoes it ([f55a332](https://github.com/CoreySpohn/eyepiece/commit/f55a33223e88fc7cd88c44b316f9db098df7ffc4))

## [0.14.0](https://github.com/CoreySpohn/eyepiece/compare/v0.13.0...v0.14.0) (2026-10-15)


### Features

* **attention:** add circumscribe, an outline that draws on around a target and fades ([9b591ba](https://github.com/CoreySpohn/eyepiece/commit/9b591ba64d1af9badb453713dfa3b8fde9efc440))
* **attention:** add indicate, a transient pulse that returns to the stage level ([2d79150](https://github.com/CoreySpohn/eyepiece/commit/2d79150ca7786a9d1a6aa24419165b49ff8aa1d6))
* **attention:** add link, a line that grows between marks in different panels ([b45d1e0](https://github.com/CoreySpohn/eyepiece/commit/b45d1e0a76e0a4fc2a83a902eff1923b0b4c1934))
* **attention:** add term_row, an equation or readout with one text per term ([d7d1929](https://github.com/CoreySpohn/eyepiece/commit/d7d19297d878fca6d7297e4a5e1112f67b472793))
* **attention:** audit leaders through an explicit check instead of raising in draw ([5e95207](https://github.com/CoreySpohn/eyepiece/commit/5e95207f8a207fa5581d9bcad319f121766d8d37))
* **attention:** fade a link by level and audit it against titles and axis labels ([223e051](https://github.com/CoreySpohn/eyepiece/commit/223e051bc9435d199a4a02f8d6e8b7948db3c12b))
* **attention:** let flight carry and turn a line segment or a bar patch ([43150e3](https://github.com/CoreySpohn/eyepiece/commit/43150e3ab01cbc7a065c10bcb947af77991b8359))


### Bug Fixes

* **attention:** fade a caption's box faster than its words ([3429a37](https://github.com/CoreySpohn/eyepiece/commit/3429a3791a0eee1c5e2dd0c895161c0b766d6b7b))

## [0.13.0](https://github.com/CoreySpohn/eyepiece/compare/v0.12.0...v0.13.0) (2026-10-15)


### Features

* **attention:** add build_axes to draw an axis on in sequence ([d03c6aa](https://github.com/CoreySpohn/eyepiece/commit/d03c6aa477b166979322e79602eb57e539e64d24))
* **attention:** add collapse to gather a row of cells into a bar ([6907feb](https://github.com/CoreySpohn/eyepiece/commit/6907feb8c4d588960b7afa984876c268fe94f68c))
* **attention:** add Stage to show, dim or hide named groups ([4803fa4](https://github.com/CoreySpohn/eyepiece/commit/4803fa4fab54d0026006e398b4aff850ea2adb38))
* **attention:** add Tempo, one factor for a staged clip's pace ([daf75f4](https://github.com/CoreySpohn/eyepiece/commit/daf75f4f57b35a3d8ddbf128b8be11307c3af85d))
* **attention:** color and weight callout key terms; let captions rise ([a877df0](https://github.com/CoreySpohn/eyepiece/commit/a877df0045b8694c7627d85cf9b49a84fbf72499))

## [0.12.0](https://github.com/CoreySpohn/eyepiece/compare/v0.11.0...v0.12.0) (2026-10-14)


### Features

* **attention:** Anchor, flight, leader and callout for explainer animations ([4c8969d](https://github.com/CoreySpohn/eyepiece/commit/4c8969d6ed9cf9d574f784ddccdfe0014408b885))

## [0.11.0](https://github.com/CoreySpohn/eyepiece/compare/v0.10.0...v0.11.0) (2026-10-12)


### Features

* **role:** resolve prior, positive, negative, posterior and truth roles ([a182899](https://github.com/CoreySpohn/eyepiece/commit/a182899790213d5ff9614953fd1f526a7e640b99))


### Bug Fixes

* **role:** lowercase hex for fixed-color fallbacks, docstring names them, test pins them to hwostyle ([13caf7b](https://github.com/CoreySpohn/eyepiece/commit/13caf7b93f2564f1a3221fca26ff8032eb78d9b1))

## [0.10.0](https://github.com/CoreySpohn/eyepiece/compare/v0.9.0...v0.10.0) (2026-10-12)


### Features

* **style:** resolve semantic color roles ([507094f](https://github.com/CoreySpohn/eyepiece/commit/507094f4a06bb1d3b2e441c46a428b04196d40c8))

## [0.9.0](https://github.com/CoreySpohn/eyepiece/compare/v0.8.0...v0.9.0) (2026-10-12)


### Features

* **marks:** add spans, ref_line and log_zero_floor ([5f92efc](https://github.com/CoreySpohn/eyepiece/commit/5f92efcd5406ab68c373656072395b9b6cb5fa72))
* **stats:** add calibration and strip diagnostics ([02cbc41](https://github.com/CoreySpohn/eyepiece/commit/02cbc41b6c4814bf2452841b668199dd0db165d5))
* **stats:** add interval_points ([05b1255](https://github.com/CoreySpohn/eyepiece/commit/05b1255ed477b15ad80e164dd9a0c51379b6a4f8))
* **stats:** add scatter_values with inset colorbar and cmap role fallback ([1792ff4](https://github.com/CoreySpohn/eyepiece/commit/1792ff41a7386f3bafdb8b7411c1d6ded25f4189))
* **style:** export neutral and backing ([2a6d569](https://github.com/CoreySpohn/eyepiece/commit/2a6d56955efaeb65e06f7aa4a9782fe2f50c9b8c))


### Bug Fixes

* **marks:** spans shades where mask is True ([4836cd3](https://github.com/CoreySpohn/eyepiece/commit/4836cd3816e1c1412234c611be23533facd6d8dc))
* **stats:** calibration keeps later data visible and log band exact; log_zero_floor pins only the floor ([de8fe2f](https://github.com/CoreySpohn/eyepiece/commit/de8fe2f4a368398865c6a1595f24c556e2d5dde4))

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
