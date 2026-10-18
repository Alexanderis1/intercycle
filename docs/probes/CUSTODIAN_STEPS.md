# Custodian steps (laptop session), split A1+

For the NEW custodian session on the LAPTOP only (PLAN_TWIN §0 machine roles, §4 Custodian row, §5). The desktop lead
and every builder lane never run these steps and never read `C:\intercycle_sealed`. The custodian does no code work and
never views, decodes or describes a test clip; it only downloads, hashes and later serves files to the test labeller.

## 1. Check the repo state

1. `git pull`, then check out the branch the lead names (it carries `prereg/split_a1.csv`).
2. Check that `prereg/ACTIVE_SPLIT` holds `split_a1.csv`, and that `prereg/split_a1.csv` hashes to
   `8c98e3ed1bb84a1e2347f65692bb5d6cd35d0fd4145ae54770f16f500f55952b` with LF line endings
   (`git show HEAD:prereg/split_a1.csv | sha256sum`; a CRLF working copy gives a different byte hash).

## 2. Download exactly the test files (about 2 GB, plain HTTPS, no HF token)

```
python scripts/custodian_fetch.py
```

- Python 3.11+ with pydantic (required: the repo's contract parser `twin.schemas.prereg`). It reads the 126 TEST rows of `prereg/split_a1.csv` and downloads each clip
  from HF `Voxel51/Safe_and_Unsafe_Behaviours` at revision `7369ff1830b5f6c00f7d7383eeeaa90704bfc5b5`
  (`https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours/resolve/<rev>/data/<clip_id>.mp4`) into
  `C:\intercycle_sealed\clips\<clip_id>.mp4`.
- It checks each file's sha256 against the split, re-downloads a wrong file, and fails if any file is missing, wrong or
  extra in that folder. It is safe to re-run: files that already verify are kept.
- On success it writes `sealed/MANIFEST.sha256` in the repo in the `twin.sealed/1` contract format (the one
  `scripts/sealed_run.py` parses): one `<sha256>  clips/<clip_id>.mp4` line per test clip, then `#ids`, `#count` and
  `#aggregate` lines. The split file's sha256 is printed, not written.
- Check it: `python scripts/custodian_fetch.py --verify-only` must print `verified 126 files`.
- Expected counts: camera 9: 114, camera 14: 12; classes 0: 51, 1: 58, 3: 9, 5: 4, 6: 1, 7: 3.
- 13 pairs of test ids are byte-identical files (same sha256, sometimes different class labels); each id is still
  downloaded to its own file. `inventory.parquet` marks them in `dup_of` / `dup_classes`.
- Commit only that file: `git add sealed/MANIFEST.sha256` then `git commit -m "custodian: MANIFEST.sha256 (A1+ test)" --
  sealed/MANIFEST.sha256`, and push it, or hand it to the lead, whichever the lead asks for.

## 3. Later steps (one line each)

- **Test labeller (U14):** spawn a fresh labeller agent inside this session that sees only the sampled test frames from
  `C:\intercycle_sealed\clips` (seeded sample list hashed first; no detector proposals, dev results or code), and writes
  its labels to `C:\intercycle_sealed\labels\`.
- **LABELS.sha256:** hash every label file (plus the frame/stroke-window sample list) into `sealed/LABELS.sha256` and
  commit only that file, before FREEZE (08:30).
- **Split freeze (USER DECISION, Sun ~07:00 CEST).** Two freezes: `prereg/FREEZE_REALCLIP` (08:30, the real-clip
  test) and `prereg/FREEZE` (~10:15, the robot part). `sealed_run.py` checks each stage against its own freeze; the
  robot stages also refuse if any real-clip input (the clips stage's import closure, `twin/schemas/labels.py`,
  `sealed/*`, the active split, the press ROI file) differs between the two.
- **Real-clip run at 09:15 (from a DETACHED WORKTREE, never from main).** The lead commits `prereg/FREEZE_REALCLIP`
  (one line: the sha of the frozen code commit) at 08:30; call the commit that writes it `<RCW>`. The custodian then
  runs, with no other change:
  `git worktree add --detach C:\intercycle_rc <RCW>` ;
  `cd C:\intercycle_rc` ;
  `python scripts/sealed_run.py check --stages clips` (refusals only) ;
  `python scripts/sealed_run.py run --stages clips` ;
  `python scripts/sealed_run.py score` (same worktree, `INTERCYCLE_DATA` set to the dev data root: inventory and the
  watchdog's TRAIN keyframes only).
  `score` is the pre-registered real-clip analysis: it re-runs the same refusals, re-hashes the attempt's PERC tables
  against RUNLOG, loads the LABELS-checked test labels (`twin.dev_labels.load_scored_test`, same status rules as dev;
  reserves by list order) and runs the dev scorer `intercycle.eval.realclip.run` with split `test`. It writes the full
  result (per-item records carry label content) to `C:\intercycle_sealed\scores\attempt_NNN_realclip_test.json` and the
  summary (figures and n only) to `results/sealed/realclip_test_attempt_NNN.json`, which the custodian commits with the
  RUNLOG. The lead reads only that summary.
  The clips stage refuses unless HEAD differs from the FREEZE_REALCLIP commit only in `prereg/FREEZE_REALCLIP` and
  `results/sealed/**`, so running it on main after later robot commits is refused by design. MANIFEST and LABELS must
  hash to their FREEZE_REALCLIP blobs. No seeds are needed. The RUNLOG it appends (`results/sealed/RUNLOG.json` in
  that worktree) may be committed there on a branch and merged into main: only `results/sealed/**` changes.
- **Seeds at the ROBOT FREEZE:** after the lead writes `prereg/FREEZE` (~10:15), derive the 16 test seeds as
  seed_i = sha256(R ‖ i), with R the randomness of the first drand round published after the FREEZE commit time
  (fallback `secrets.token_hex(16)`, disclosed), and commit `prereg/TEST_SEED_COMMIT` = sha256(salt ‖ seeds); then run
  the robot stages (`run --stages double` / `isaac`, once `scripts/sealed_robot.py` is wired) when the lead requests it.

## Test clips (from `prereg/split_a1.csv`; ids, sha256, group and framing only)

split_a1.csv sha256 (LF line endings, as committed): `8c98e3ed1bb84a1e2347f65692bb5d6cd35d0fd4145ae54770f16f500f55952b`  
Test clips: 126 (camera 9: 114, camera 14: 12); per class: 0: 51, 1: 58, 3: 9, 5: 4, 6: 1, 7: 3.

| # | clip_id | sha256 | group | framing |
|---|---|---|---|---|
| 1 | 0_te7 | `8bad1dd46d4554d4303e84caea3579c15e25d490e8c5b26ff9008900675a1f40` | c9_a007 | c9_f1 |
| 2 | 0_te8 | `f6b5371af35e9fe3ed010bc9f287fe3dd9f3a3d8a46d60ff0ce2e624abb24ab9` | c9_a007 | c9_f1 |
| 3 | 0_te9 | `08417650b38fb354164da00d170f2343fac2244df2f0f9d9dae724135663e47f` | c9_a007 | c9_f1 |
| 4 | 0_tr15 | `de8645c347843669a7e9cb6ffb78004531f346d87dc2d86dcac39fa08209a619` | c9_a007 | c9_f1 |
| 5 | 0_tr16 | `31be68f203b8fa5895b5f8566b84d5d8407636113fd34d134983cc4962ea1c32` | c9_a007 | c9_f1 |
| 6 | 0_tr17 | `81b2d25247a24ac23ba30b744159c134928d75e4a741bd339f0e6277a662ac72` | c9_a007 | c9_f1 |
| 7 | 0_tr18 | `1e1966d237683fb484f0af9b96b62ec4aa14a1993f994624100bca87c7629fe8` | c9_a007 | c9_f1 |
| 8 | 0_tr19 | `08918de31a92d4103989aadd70fd6076b5d1cb1c60dc67c842c3999bbc5cf9ca` | c9_a007 | c9_f1 |
| 9 | 0_tr45 | `2b12fb4398de118cd26129431a62326552c95937774562d51d79dcfb510df4ca` | c9_a014 | c9_f1 |
| 10 | 0_tr46 | `a1d971a38c11e70f7590242ce5a0494fa13a743e4758fb86744eb8e34c3d07d6` | c9_a014 | c9_f1 |
| 11 | 0_tr48 | `98c42c589bbfb892a4b1ad7969205b1569ff1fda14933a34348e4906e4b29525` | c9_a014 | c9_f1 |
| 12 | 0_tr49 | `205b340d01acf3d1c6f7fdf229dca3bbbf668e66e85c6ea8bf83ba6d6835593f` | c9_a014 | c9_f1 |
| 13 | 0_tr50 | `ca44e8d9e6ea6d0ab73ae1f0cf61e020c160e0cd2ca5531284a3783eb67b5f2f` | c9_a014 | c9_f1 |
| 14 | 0_tr51 | `9bb024df8e7d19b6cf4d81f7f4f81ab269e0c26cef3e8334cbc6086b3725013c` | c9_a014 | c9_f1 |
| 15 | 0_tr52 | `2155fbb4eeb51c340e9c6a8a4d8509030daa546f479ec9c9f4988d414359976c` | c9_a014 | c9_f1 |
| 16 | 0_tr53 | `7b90edaa5b903b5aa42be128335e74e8914d13e3c4817ee6f72bd9c89709c596` | c9_a014 | c9_f1 |
| 17 | 0_tr54 | `9b357138e5498861a96a587d3a5afacfd1f2c560776e054022084c5cce12aed6` | c9_a014 | c9_f1 |
| 18 | 0_tr55 | `8059b8586575e85455da0245b212bf0770dd7026cf22187e4dc0740536e9cd5b` | c9_a014 | c9_f1 |
| 19 | 0_tr56 | `255ac9cbaa09d30b8f4d2c7b5fef80f6b178960631081ec9a86bed40063b136c` | c9_a014 | c9_f1 |
| 20 | 0_tr57 | `cfd70074080f4e4491ff1821d09d13c239dc53ba7de81de0e32762f081cf3abf` | c9_a014 | c9_f1 |
| 21 | 0_tr58 | `bd55ad4de3455aa764a4dbeae68858ee44a05c81bbf67d38bd78adfbd367128d` | c9_a014 | c9_f1 |
| 22 | 0_tr59 | `9da2b0339c523530ab42d9ba3cd90d06bbd9ebd9283c073f415762afc30eed18` | c9_a014 | c9_f1 |
| 23 | 0_tr60 | `dfadc40b3d4965c7e7959e9fdb5f14972b687f63d07d4959adffe21cafd228f5` | c9_a014 | c9_f1 |
| 24 | 0_tr61 | `db511dd922e9c6550120dbd6c2bf924d7d4a3910226182f7f902fae570c495ba` | c9_a014 | c9_f1 |
| 25 | 0_tr62 | `bf43a46cd72a9427d196a3d9b6baa0c57b34056cf46d13cdf3625694cb5c7e82` | c9_a014 | c9_f1 |
| 26 | 0_tr63 | `b37118cf20d96d4607f55ca29aa437394537563ce6e17956855dc2ffa8a0f9fa` | c9_a014 | c9_f1 |
| 27 | 0_tr64 | `66fe8131a647154dac7a26e6fda9f2d842fdb8f77432e9e2a4c7002b9d1c26fb` | c9_a014 | c9_f1 |
| 28 | 0_tr65 | `fda2c69a6a6327fb159f8fcc76cd353b01446758308dfdfe7141ee8a44eb4625` | c9_a014 | c9_f1 |
| 29 | 0_tr66 | `e581a05dae0b21969dd2f291cc6ffb2233928333e33891ae34e6dfd1a69b595e` | c9_a014 | c9_f1 |
| 30 | 0_tr67 | `271d41e36785a8049d9c9c79512e459562b8b36d77a1c1ecaeb0a5da74a77bbd` | c9_a014 | c9_f1 |
| 31 | 0_tr68 | `79949f3a544ca4ea0a06836282233ac9015f85a6a6c4b1dc7f3f1151e740d9c2` | c9_a014 | c9_f1 |
| 32 | 0_tr69 | `974ae2e5136de5bb6eafb3240edbcc08d1d70896f290eee4e79b79564a91faaf` | c9_a014 | c9_f1 |
| 33 | 0_tr70 | `7b1e51c6c66ca4de0789917ba4976374fbf5b8f149800a8a11f2bf5fb0f8c884` | c9_a014 | c9_f1 |
| 34 | 0_tr71 | `1028fd8dd208b3b4fd040cd38c83b8526c9a4cf7bcf9cbeb95979e4c9a46d860` | c9_a007 | c9_f1 |
| 35 | 0_tr72 | `6543c8d988f173ca7fa7a3de999ed9cecbd02670bfd02b6ccbb94dde4a5f43a3` | c9_a007 | c9_f1 |
| 36 | 0_tr73 | `1b8cb84015e3e38404f1fcd1c2066f1ea77227bb3f0c0cb2664c977e00749924` | c9_a007 | c9_f1 |
| 37 | 0_tr74 | `63ade6067285ab8e6be730e7830c98878179840a6e9968a9414ba48a8c5537df` | c9_a007 | c9_f1 |
| 38 | 0_tr75 | `0ed9178fe9a5dea63d2faecec6936158cb45452e7029b6f34255ac10595322e0` | c9_a007 | c9_f1 |
| 39 | 0_tr79 | `3d5789ab5b966b443c17abcbd0585008162356dd869ca095cd7f6994fbdb2701` | c9_a014 | c9_f1 |
| 40 | 0_tr80 | `a739d2f38de66246bbb374c615a0d541e8caf9888a1bac24cfba9d84a7ab544d` | c9_a014 | c9_f1 |
| 41 | 0_tr81 | `a469f0810d98c08411f8f940f9acb773927b179e8a62ab5b3428c26a4f03e5b0` | c9_a014 | c9_f1 |
| 42 | 0_tr82 | `de0d97f32229b3b8a709bae71911c0b11dc6bcc0d18d04c207e78b2c0396386a` | c9_a014 | c9_f1 |
| 43 | 0_tr83 | `67439f60d10ef3db4c9080d6ec78ef6bd9dce6606a1cd9c6b75389c0fffe95db` | c9_a014 | c9_f1 |
| 44 | 0_tr84 | `cc58b735cefba1e0356ae02e15febc90fb06cd91d6ff149389dc1719d1fac793` | c9_a014 | c9_f1 |
| 45 | 0_tr85 | `f19c3cda098090f3e3eeec130156595dfb49c88896a6ecd1d1f97619463b7580` | c9_a014 | c9_f1 |
| 46 | 0_tr86 | `6526e93f36396fe7ce41a48f795ca6b4f2189d0254b5d3e39fc4245d4dc6bfe1` | c9_a014 | c9_f1 |
| 47 | 0_tr87 | `63cb8d7264112f3e311e11e1c4368891820c6537a3e78f939b712af54557b6e4` | c9_a014 | c9_f1 |
| 48 | 0_tr88 | `41d3ab230b9a185b5d20e4b8e42c65ba813229f4a3597116efbab6c29d9c1990` | c9_a014 | c9_f1 |
| 49 | 0_tr89 | `fd28693ff0c2174a069f6cc8afd6d7d0a65256f92ef7f297b3f104a36856f21b` | c9_a014 | c9_f1 |
| 50 | 0_tr90 | `43f2c90ba106b698e3703a6d09ec9f2a40e722ac3a38c7fcaa6652c9d06dc707` | c9_a014 | c9_f1 |
| 51 | 0_tr91 | `a1c8253fd21c890e86d1838e612d2053348bb08893729082924e3c64b9f11e60` | c9_a014 | c9_f1 |
| 52 | 1_te1 | `f6b5371af35e9fe3ed010bc9f287fe3dd9f3a3d8a46d60ff0ce2e624abb24ab9` | c9_a007 | c9_f1 |
| 53 | 1_te11 | `c0bc265f24fc1b970a1cb49f2050ca092857e4f4c987f71d8f717d90bdf455e8` | c9_a017 | c9_f1 |
| 54 | 1_te2 | `263ff83b2e505bb436b9f5711b8d1181798cfb3123ef65c586b6e32370441c31` | c9_a007 | c9_f1 |
| 55 | 1_te3 | `b916aafb0b8702109c5a1fb15a927c499cd26c3cc05a8b1239c68f7e773af152` | c9_a007 | c9_f1 |
| 56 | 1_te6 | `3e7db8b7bbf61810dbf20ac24ff01971f190885c866fa91be1feedf077ef9669` | c9_a017 | c9_f1 |
| 57 | 1_tr1 | `2155fbb4eeb51c340e9c6a8a4d8509030daa546f479ec9c9f4988d414359976c` | c9_a014 | c9_f1 |
| 58 | 1_tr10 | `1be022b122fa264df897f27bd1c91ff4efd7f4d94e1b528ef4ee700f4b53c505` | c9_a014 | c9_f1 |
| 59 | 1_tr11 | `bf43a46cd72a9427d196a3d9b6baa0c57b34056cf46d13cdf3625694cb5c7e82` | c9_a014 | c9_f1 |
| 60 | 1_tr12 | `b37118cf20d96d4607f55ca29aa437394537563ce6e17956855dc2ffa8a0f9fa` | c9_a014 | c9_f1 |
| 61 | 1_tr13 | `e1e0a5538d045d0f1d8a254bd5aed27b91f37ea7587454f7427b84a116b0642e` | c9_a014 | c9_f1 |
| 62 | 1_tr14 | `66fe8131a647154dac7a26e6fda9f2d842fdb8f77432e9e2a4c7002b9d1c26fb` | c9_a014 | c9_f1 |
| 63 | 1_tr15 | `271d41e36785a8049d9c9c79512e459562b8b36d77a1c1ecaeb0a5da74a77bbd` | c9_a014 | c9_f1 |
| 64 | 1_tr16 | `8f2ac42eb2dcf3acda2dc0d9981f39433271034bef039f2c6417bf8333aaf4b6` | c9_a014 | c9_f1 |
| 65 | 1_tr17 | `7d99ccc4a3166e13a63b0efe37fd93ff6f4a8b075185fe416f2462c7caa2c756` | c9_a014 | c9_f1 |
| 66 | 1_tr18 | `79949f3a544ca4ea0a06836282233ac9015f85a6a6c4b1dc7f3f1151e740d9c2` | c9_a014 | c9_f1 |
| 67 | 1_tr19 | `7b706d27be01e68d6d9f808ae778a68ddfde31116a89e4e563c08b275ebd10ba` | c9_a007 | c9_f1 |
| 68 | 1_tr2 | `c89c6b373a3e4cb6cc3652cc7c5a7cd24ae52218ffca751eccfcae13e145609c` | c9_a014 | c9_f1 |
| 69 | 1_tr20 | `73971f20ec0769b9b784ac78597ec6044ada6d8521c11b81fdec8e4ab87f7bed` | c9_a007 | c9_f1 |
| 70 | 1_tr21 | `fbabe84410ff6c39486fe5b49f055ea76be148d2db6191e8ed79f1fe891a721c` | c9_a007 | c9_f1 |
| 71 | 1_tr22 | `8aaf58e3252d1460300bc0261be823de07f4e0c51c3fc5663194fc1da65482bc` | c9_a007 | c9_f1 |
| 72 | 1_tr23 | `6be8664b0fcd51375440637da0fb5bf121566eb3964b99dbfc43365a1d413e5b` | c9_a007 | c9_f1 |
| 73 | 1_tr24 | `8bad1dd46d4554d4303e84caea3579c15e25d490e8c5b26ff9008900675a1f40` | c9_a007 | c9_f1 |
| 74 | 1_tr25 | `2f908248a6116c2d8b43cab7b7053289c2df04581fd0ffeb23dd41764c9c5cac` | c9_a007 | c9_f1 |
| 75 | 1_tr26 | `8d0f92c67b212ae0b4e3bf8412f2b12fa7962fee180e676f00d6520f4c2f8c8e` | c9_a017 | c9_f1 |
| 76 | 1_tr27 | `2fd6cbb8d7711c0b51c1390ec43270752e2e2f2e1ac33bd46996ae29523df4ff` | c9_a007 | c9_f1 |
| 77 | 1_tr28 | `e56bd0b6924c09f3ceb46f8f688ae35d762a7bc3cf88a958b38cd49456d0bfc9` | c9_a007 | c9_f1 |
| 78 | 1_tr3 | `7cc6e0b2b8f9ee4832f0f7e63f500b35dd225aa66b8ad8ea620fb66a37ad2d23` | c9_a014 | c9_f1 |
| 79 | 1_tr30 | `b34ba15968cd2c72c65f006a41d08de5ddcc5ec24bb063c858128897d4c1bd44` | c9_a017 | c9_f1 |
| 80 | 1_tr31 | `b6f11945d81ce4dd9163e28628aed0f2fb14ad841cd4d38d1acd333cee279a0e` | c9_a007 | c9_f1 |
| 81 | 1_tr35 | `d849da5b21e9b71c18f082266604e674e7e57b27fa94a63a4030d14ebdff5f3b` | c9_a017 | c9_f1 |
| 82 | 1_tr37 | `3b10d054c4d98947fc682b48219c9643793e30f887ce5c4033b05dc7ebbb2e4d` | c9_a017 | c9_f1 |
| 83 | 1_tr4 | `e5f52ce112d1f432c774e2f949fb4f910faf6a1c8085e8cf034fab3b3ea5d074` | c9_a014 | c9_f1 |
| 84 | 1_tr49 | `cbea184b811ba3cf2cc4bf60c713e110e7d65de03495f33fde533bba803c428d` | c9_a017 | c9_f1 |
| 85 | 1_tr5 | `bd55ad4de3455aa764a4dbeae68858ee44a05c81bbf67d38bd78adfbd367128d` | c9_a014 | c9_f1 |
| 86 | 1_tr50 | `7a5bda2db747e001aa7f1b88b8d0e509ee02d3a513e2ac841591c68781d3f278` | c9_a017 | c9_f1 |
| 87 | 1_tr6 | `9da2b0339c523530ab42d9ba3cd90d06bbd9ebd9283c073f415762afc30eed18` | c9_a014 | c9_f1 |
| 88 | 1_tr68 | `bd178f454f749508bccfaedadc5136ec08583a88155a3c3bd91e4234d7559c32` | c9_a017 | c9_f1 |
| 89 | 1_tr69 | `fa471989117d98391a413e6796e52e831198f90fa92810dc44d97a58be56056b` | c9_a017 | c9_f1 |
| 90 | 1_tr7 | `db511dd922e9c6550120dbd6c2bf924d7d4a3910226182f7f902fae570c495ba` | c9_a014 | c9_f1 |
| 91 | 1_tr70 | `e9af2c721a96f4ed94f20189236ca185743d81a490057893499cc4a641f79efb` | c9_a017 | c9_f1 |
| 92 | 1_tr71 | `c02b67425af75cd0cb964db2223c7a13f8df977a3465bec6d18474daec8f9112` | c9_a017 | c9_f1 |
| 93 | 1_tr72 | `d9e0aecfa1615769477b5a8c605f290185ab740f208a2fead010d95b925d7388` | c9_a017 | c9_f1 |
| 94 | 1_tr79 | `af2270ff295f8373429208655a715c1384ba506318a2a33c6430846dfe55e07b` | c9_a017 | c9_f1 |
| 95 | 1_tr8 | `f03019498b88a43923d78ef8e4edcd906c5c0b2c7ea3c9afad79b080fc2f6f82` | c9_a014 | c9_f1 |
| 96 | 1_tr80 | `b5d8e4cba64c88f48f99cb40074e77ce7a70aa55987483849f230c6f3cee7e61` | c9_a017 | c9_f1 |
| 97 | 1_tr82 | `2c2ab43e3fb193cabd1636c6d6c7aee7c65f6d30e3c650c4bf6967759213dee4` | c9_a017 | c9_f1 |
| 98 | 1_tr83 | `734371a56b03d264487e34c573eeda9d1076275abcb32a18e8205d04833a8c0f` | c9_a017 | c9_f1 |
| 99 | 1_tr84 | `4905adb9cb75aa4422945ee176b9016cb84ce0de4d853a646e47e0d083af5978` | c9_a017 | c9_f1 |
| 100 | 1_tr85 | `62911f198e5455383c262def970a7c4a2f8de1d4dbc5719ac57d11e840554e38` | c9_a014 | c9_f1 |
| 101 | 1_tr86 | `56149e451621242213c9abf1beb180fbdfd71c78def6e26ad23ebb95cb850d50` | c9_a007 | c9_f1 |
| 102 | 1_tr87 | `0c39d940dbef59259d83701dc1e5cfd5b5cb8d1d0c966ac058f75ce97c64488c` | c9_a007 | c9_f1 |
| 103 | 1_tr9 | `da853813f5aebbb5500dfc4bf952346055a2970ffcddcd93c4ac4749e3be634f` | c9_a014 | c9_f1 |
| 104 | 1_tr91 | `98c42c589bbfb892a4b1ad7969205b1569ff1fda14933a34348e4906e4b29525` | c9_a014 | c9_f1 |
| 105 | 1_tr93 | `fc15b7620aa8d05f7889694f8a2205786fb0dfd6d00dd71b84239258d8d8ed3c` | c9_a014 | c9_f1 |
| 106 | 1_tr94 | `1e39f568871fdc078f9ed7653690df2f5a2d46ddfbeca98be4ab78c0282a188b` | c9_a014 | c9_f1 |
| 107 | 1_tr95 | `eb9231b53004d9373e27b97e161c1097765f807f3f717b6eb4fc81e0bc10d4cb` | c9_a014 | c9_f1 |
| 108 | 1_tr96 | `8348e678e78abfdfff32abca59cb8ebd8ff374a9ef26f7b93027ea9b39f2dc0b` | c9_a014 | c9_f1 |
| 109 | 1_tr97 | `9bb024df8e7d19b6cf4d81f7f4f81ab269e0c26cef3e8334cbc6086b3725013c` | c9_a014 | c9_f1 |
| 110 | 3_te3 | `612ec2bd165c5777a5282b3778c1df4deebedeafbea8be762ec68cac825e5d26` | c14_a002 | c14_f0 |
| 111 | 3_te7 | `be04d7b7664b890026cb54bda78d2d0fb29c613bbe4f730fdca3b834bcd24d19` | c14_a004 | c14_f0 |
| 112 | 3_tr15 | `f8068e102a1f929a94c35892d67b0bf3eb9aa01b8127fea8dff0ff270d09b4e0` | c14_a007 | c14_f0 |
| 113 | 3_tr17 | `4f19c583011b673481877b556476146ca9c1fca3126ba035db5f0d52c0935320` | c14_a008 | c14_f0 |
| 114 | 3_tr19 | `fcfeaafbc86acce9a7374d6b774df75c39a9a4abc02e5ff3871753ffa13e498c` | c14_a009 | c14_f0 |
| 115 | 3_tr29 | `74f9bd4518daac7ed9eebfa24aad255f807460056c62ddae952b6922ef254900` | c14_a015 | c14_f0 |
| 116 | 3_tr37 | `c33a5103766ac79f6a3223b5d785af0ea611565dbe3a902268e36f8145235f89` | c14_a019 | c14_f0 |
| 117 | 3_tr42 | `c8e61b5634536a5f498c5c9a9246bf6ebc1e4bda37bfd79564b4e45766d061f2` | c14_a022 | c14_f0 |
| 118 | 3_tr43 | `e4edf2affe7fc44069aaf4a4a27ba963e73beafb5abb912a8afd036cc1bdc256` | c14_a023 | c14_f0 |
| 119 | 5_te15 | `af1df2c5f05e707b9232bb791c936956b3f6a8df8d6b164fbe54c449ca927a04` | c9_a017 | c9_f1 |
| 120 | 5_tr21 | `87ea099c168130adb04d9407359215b9873179bc9baf416f6d0cad161415172f` | c9_a017 | c9_f1 |
| 121 | 5_tr22 | `a453ed3d563e2382d7f1edd3240bb507dbaf4b5f98117d314812079ecaaa6d9a` | c9_a017 | c9_f1 |
| 122 | 5_tr23 | `d7a1069288fe03631fea5de01245e4bbf439da83de9d95e12e04f4b372cdd31b` | c9_a017 | c9_f1 |
| 123 | 6_tr12 | `3a537d9a358de52ba8c36b63ee0f9404a1e458a7a3da2e4742db610b76446e41` | c9_a020 | c9_f0 |
| 124 | 7_te5 | `37a357cf69ac970ee369831ad9c3b5245e56091c5a0466ed41b88d215584f6cc` | c14_a028 | c14_f0 |
| 125 | 7_tr1 | `86f301485074d22fd1df5024ec53d1f293d40b2bcdc374fbeb20fbe38e468cb8` | c14_a029 | c14_f0 |
| 126 | 7_tr14 | `3f7d8f45a980c50228a04536c07d9a9c059286d796c95456c26010f00cb9ff3d` | c14_a033 | c14_f0 |
