# Third-party test content

## Quaternius animation packs (CC0)

- Universal Animation Library: https://quaternius.itch.io/universal-animation-library
- Universal Animation Library 2: https://quaternius.itch.io/universal-animation-library-2
- Author: Quaternius. License: Creative Commons Zero v1.0 Universal
  (free for personal, educational and commercial projects; attribution
  is not required but given here with thanks).

Files: only `starter/ual1_standard.glb` (7.6 MB, no-root-motion pack 1)
is vendored as the animation showcase fixture (regression test
`starter_pack_playback_moves_joints`, File → Load entry point).
Full archives stay out — download the free `Standard.zip` packs from
the links above. Verified 2026-10-01: 43/43 clips assemble, 0 skipped,
all LINEAR — see `cargo run -p ornis-gltf --example inspect -- <file>`.
