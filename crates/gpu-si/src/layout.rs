//! Word offsets of the contact batch record.
//!
//! The batch is a `#[repr(C, align(16))]` SoA record — every field is a
//! four-lane array, so field `i` starts at word `4 * i` — and the kernel reads
//! it as words. Each constant below is the word offset of the same-named field
//! (lane `l` of `nx` is word [`NX`] `+ l`). `ornis-physics` pins every one of
//! them against `offset_of!(GpuBatch, …)`, so the runtime layout and this table
//! cannot drift apart.
//!
//! Only the groups the kernel loads are listed. The packer fills a few more
//! (the precomputed `ra×n` / `ra×t` factors); they are not here because nothing
//! reads them on the device, and a constant no shader uses is a claim nobody
//! checks. Reordering the record still moves a listed group, so the gate sees
//! it.

/// Words per contact batch record, including the trailing `count` and padding.
pub const BATCH_WORDS: u32 = 312;

/// Words of the `nx` lane group.
pub const NX: u32 = 0;

/// Words of the `ny` lane group.
pub const NY: u32 = 4;

/// Words of the `nz` lane group.
pub const NZ: u32 = 8;

/// Words of the `rax` lane group.
pub const RAX: u32 = 12;

/// Words of the `ray` lane group.
pub const RAY: u32 = 16;

/// Words of the `raz` lane group.
pub const RAZ: u32 = 20;

/// Words of the `rbx` lane group.
pub const RBX: u32 = 24;

/// Words of the `rby` lane group.
pub const RBY: u32 = 28;

/// Words of the `rbz` lane group.
pub const RBZ: u32 = 32;

/// Words of the `t1x` lane group.
pub const T1X: u32 = 60;

/// Words of the `t1y` lane group.
pub const T1Y: u32 = 64;

/// Words of the `t1z` lane group.
pub const T1Z: u32 = 68;

/// Words of the `t2x` lane group.
pub const T2X: u32 = 72;

/// Words of the `t2y` lane group.
pub const T2Y: u32 = 76;

/// Words of the `t2z` lane group.
pub const T2Z: u32 = 80;

/// Words of the `inv_ma` lane group.
pub const INV_MA: u32 = 132;

/// Words of the `inv_mb` lane group.
pub const INV_MB: u32 = 136;

/// Words of the `inv_k_n` lane group.
pub const INV_K_N: u32 = 144;

/// Words of the `inv_k_t1` lane group.
pub const INV_K_T1: u32 = 148;

/// Words of the `inv_k_t2` lane group.
pub const INV_K_T2: u32 = 152;

/// Words of the `apply_n_ax` lane group.
pub const APPLY_N_AX: u32 = 156;

/// Words of the `apply_n_ay` lane group.
pub const APPLY_N_AY: u32 = 160;

/// Words of the `apply_n_az` lane group.
pub const APPLY_N_AZ: u32 = 164;

/// Words of the `apply_n_bx` lane group.
pub const APPLY_N_BX: u32 = 168;

/// Words of the `apply_n_by` lane group.
pub const APPLY_N_BY: u32 = 172;

/// Words of the `apply_n_bz` lane group.
pub const APPLY_N_BZ: u32 = 176;

/// Words of the `apply_w_ax` lane group.
pub const APPLY_W_AX: u32 = 180;

/// Words of the `apply_w_ay` lane group.
pub const APPLY_W_AY: u32 = 184;

/// Words of the `apply_w_az` lane group.
pub const APPLY_W_AZ: u32 = 188;

/// Words of the `apply_w_bx` lane group.
pub const APPLY_W_BX: u32 = 192;

/// Words of the `apply_w_by` lane group.
pub const APPLY_W_BY: u32 = 196;

/// Words of the `apply_w_bz` lane group.
pub const APPLY_W_BZ: u32 = 200;

/// Words of the `w_a00` lane group.
pub const W_A00: u32 = 204;

/// Words of the `w_a01` lane group.
pub const W_A01: u32 = 208;

/// Words of the `w_a02` lane group.
pub const W_A02: u32 = 212;

/// Words of the `w_a10` lane group.
pub const W_A10: u32 = 216;

/// Words of the `w_a11` lane group.
pub const W_A11: u32 = 220;

/// Words of the `w_a12` lane group.
pub const W_A12: u32 = 224;

/// Words of the `w_a20` lane group.
pub const W_A20: u32 = 228;

/// Words of the `w_a21` lane group.
pub const W_A21: u32 = 232;

/// Words of the `w_a22` lane group.
pub const W_A22: u32 = 236;

/// Words of the `w_b00` lane group.
pub const W_B00: u32 = 240;

/// Words of the `w_b01` lane group.
pub const W_B01: u32 = 244;

/// Words of the `w_b02` lane group.
pub const W_B02: u32 = 248;

/// Words of the `w_b10` lane group.
pub const W_B10: u32 = 252;

/// Words of the `w_b11` lane group.
pub const W_B11: u32 = 256;

/// Words of the `w_b12` lane group.
pub const W_B12: u32 = 260;

/// Words of the `w_b20` lane group.
pub const W_B20: u32 = 264;

/// Words of the `w_b21` lane group.
pub const W_B21: u32 = 268;

/// Words of the `w_b22` lane group.
pub const W_B22: u32 = 272;

/// Words of the `bias` lane group.
pub const BIAS: u32 = 276;

/// Words of the `spec_target` lane group.
pub const SPEC_TARGET: u32 = 280;

/// Words of the `mu` lane group.
pub const MU: u32 = 284;

/// Words of the `body_a` lane group.
pub const BODY_A: u32 = 288;

/// Words of the `body_b` lane group.
pub const BODY_B: u32 = 292;

/// Words of the `acc` lane group.
pub const ACC: u32 = 296;

/// Words of the `acc_f1` lane group.
pub const ACC_F1: u32 = 300;

/// Words of the `acc_f2` lane group.
pub const ACC_F2: u32 = 304;
