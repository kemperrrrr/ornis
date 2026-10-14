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
