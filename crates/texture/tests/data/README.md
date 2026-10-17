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

## Bake fixtures (issue #129)

`bake-gradient.png` is a 16×16 RGBA8 ramp generated for the offline bake
tests: red `40 + x*8`, green `30 + y*6`, blue `90 + x*2 + y`, alpha `210 + y`.
`bake-etc1s.ktx2` and `bake-uastc.ktx2` are that PNG encoded by the reference
`basisu` from Basis Universal tag `v2_1_0` (commit
`45d5f41015eecd9570d5a3f89ab9cc0037a25063`). `basisu -version` reports
`Basis Universal LDR/HDR GPU Texture Supercompression System v2.10.0`.

Both commands are the pinned BaseColor argv (`-srgb -mip_srgb`, Kaiser mips,
clamp). The build had SSE off, Zstandard on, OpenCL off.

```text
basisu -ktx2 -mipmap -output_file bake-etc1s.ktx2 -etc1s -q 128 -srgb -mip_srgb -mip_clamp -mip_filter kaiser bake-gradient.png
basisu -ktx2 -mipmap -output_file bake-uastc.ktx2 -uastc -uastc_level 2 -srgb -mip_srgb -mip_clamp -mip_filter kaiser bake-gradient.png
basisu -ktx2 -mipmap -output_file bake-normal-uastc.ktx2 -uastc -uastc_level 2 -normal_map -linear -mip_linear -mip_renorm -mip_clamp -mip_filter kaiser bake-gradient.png
```

Two runs of each command produced identical bytes. The normal-map command is
the pinned Normal argv (linear DFD, not the sRGB DFD of the other two). The
build had SSE off, Zstandard on, OpenCL off — `basisu -version` also prints
`No SSE, Multithreading: 1, Zstandard support: 1, OpenCL: 0`. Transcoding
level 0 back to RGBA8 stays within 22 per channel for ETC1S and 5 per channel
for UASTC. The normal-map tolerance is recorded next to the bake test.
These files are inputs to `ornis-assets` bake tests; they are not a second
copy of the 40×40 corpus above.
