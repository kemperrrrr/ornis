# KTX2 fixtures

`etc1s.ktx2` and `uastc.ktx2` are the 40×40, six-level fixtures from
[marcogomez/basisu](https://github.com/marcogomez/basisu/tree/cca5a1c8f90147a5f2b74bcabbc6329d30c9d6f2/basisu/tests/fixtures),
commit `cca5a1c8f90147a5f2b74bcabbc6329d30c9d6f2` (Apache-2.0; see
`LICENSE-APACHE`). Their bytes are unchanged.

`uastc-zstd.ktx2` is a derived fixture: each original UASTC mip was compressed
independently with Zstandard level 3, supercompression set to 2 and the six
level-index entries updated. Dimensions, DFD, metadata and decoded blocks are
unchanged. It tests Zstd decompression rather than a second image.

The tests also derive native BC7/ASTC 4×4/ETC2 RGBA containers in memory to
verify byte-preserving uploads and RGBA fallback with the same mip chain.

Review-regression fixtures are derived from `uastc.ktx2` with the same license:

- `uastc-30.ktx2`: 30×30, five levels. Each authored 40×40-chain level's
  block grid is cropped at the right/bottom to `ceil(max(30 >> mip,1)/4)`;
  the KTX2 header and level index are rebuilt with the original UASTC DFD.
- `bc7-{30,40}.ktx2`, `astc-{30,40}.ktx2`, `etc2-{30,40}.ktx2`: the respective
  Basis chains transcoded with `DecodeFlags::HIGH_QUALITY`, using `basisu`
  0.1.0 / `ornis-texture`. Native DFDs come from `ktx2::dfd::Basic::from_format`;
  levels are stored tail first and aligned to 16 bytes. They exercise GPU
  feature fallback and unaligned-base fallback, not just Basis target selection.
- `uastc-alpha.ktx2`: unchanged block payload with the DFD UASTC channel set
  to RGBA (channel ID 3). Alpha is opaque, but semantically present, so BC1
  selection must conservatively use RGBA8 and retain all authored levels.

See also `docs/material-textures-changelog.md` for the import API migration.
