//! Raw C ABI of the Keryx inference engine.
//!
//! This crate mirrors `native/include/keryx_infer.h`. It contains no model logic. The safe
//! `keryx-infer` crate loads the shared library dynamically; a miner that links statically can
//! enable the `link` feature.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

pub const C_ABI: u32 = 2;
pub const ENGINE_ABI: u32 = 9;
pub const ARCH_MAX: usize = 64;

pub const BACKEND_CUDA: u16 = 1;
pub const BACKEND_VULKAN: u16 = 2;
pub const BACKEND_SYCL: u16 = 3;
pub const BACKEND_METAL: u16 = 4;
pub const BACKEND_CPU: u16 = 5;

pub const VENDOR_NVIDIA: u16 = 1;
pub const VENDOR_AMD: u16 = 2;
pub const VENDOR_INTEL: u16 = 3;
pub const VENDOR_APPLE: u16 = 4;
pub const VENDOR_OTHER: u16 = 255;

pub const UNIT_SM: u8 = 1;
pub const UNIT_CU: u8 = 2;
pub const UNIT_XE_CORE: u8 = 3;
pub const UNIT_GPU_CORE: u8 = 4;
pub const UNIT_THREAD: u8 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardClass {
    pub backend: u16,
    pub vendor: u16,
    pub arch: [c_char; ARCH_MAX],
    pub planned_units: u16,
    pub unit_kind: u8,
}

impl Default for CardClass {
    fn default() -> Self {
        Self { backend: 0, vendor: 0, arch: [0; ARCH_MAX], planned_units: 0, unit_kind: 0 }
    }
}

#[repr(C)]
pub struct Model {
    _private: [u8; 0],
}

pub type OnToken = extern "C" fn(user: *mut c_void, token: i32) -> c_int;
pub type OnRows = extern "C" fn(user: *mut c_void, t0: i32, n_rows: i32, n_ubatch: i32, hashes: *const u8) -> c_int;

#[cfg_attr(feature = "link", link(name = "keryx_infer"))]
extern "C" {
    pub fn keryx_infer_c_abi() -> c_int;
    pub fn keryx_infer_engine_abi() -> c_int;
    pub fn keryx_infer_version() -> *const c_char;
    pub fn keryx_infer_last_error() -> *const c_char;

    pub fn keryx_infer_plan_units(backend: u16, arch: *const c_char, real_units: c_int) -> c_int;
    pub fn keryx_infer_card_for_device(backend: u16, device: c_int, out: *mut CardClass) -> c_int;
    pub fn keryx_infer_manifest_patch_set(backend: u16, out: *mut u8) -> c_int;
    pub fn keryx_infer_compat_id(cards: *const CardClass, n_cards: usize, out: *mut u8) -> c_int;

    pub fn keryx_infer_load_bound(
        gguf_path: *const c_char,
        gpu: c_int,
        blob: *mut c_void,
        held: usize,
        cut: usize,
        layout_len: usize,
        n_ctx: c_int,
    ) -> *mut Model;
    pub fn keryx_infer_context_acquire(model: *mut Model, n_ctx: c_int) -> c_int;
    pub fn keryx_infer_context_release(model: *mut Model);

    pub fn keryx_infer_trace_init(
        model: *mut Model,
        manifest: *const u8,
        len: usize,
        n_layers: u32,
        n_used: u32,
        n_attn: u32,
        n_recur: u32,
        seq_heads: u32,
    ) -> c_int;
    pub fn keryx_infer_trace_leaves(model: *mut Model) -> u32;
    pub fn keryx_infer_trace_seq_leaves(model: *mut Model) -> u32;

    pub fn keryx_infer_generate_traced(
        model: *mut Model,
        prompt: *const i32,
        n_prompt: c_int,
        max_tokens: c_int,
        on_token: OnToken,
        on_rows: OnRows,
        user: *mut c_void,
        seq_out: *mut u8,
        seq_cap: usize,
    ) -> c_int;

    pub fn keryx_infer_replay(
        model: *mut Model,
        tokens: *const i32,
        n_pos: c_int,
        n_prompt: c_int,
        pos_wants: *const i64,
        n_pos_wants: c_int,
        pos_out: *mut u8,
        pos_bytes: usize,
        seq_wants: *const i64,
        n_seq_wants: c_int,
        seq_out: *mut u8,
        seq_bytes: usize,
        on_rows: OnRows,
        user: *mut c_void,
        seq_hashes: *mut u8,
        seq_hashes_cap: usize,
    ) -> c_int;

    pub fn keryx_infer_tokenize(
        model: *mut Model,
        text: *const c_char,
        len: c_int,
        add_special: c_int,
        parse_special: c_int,
        out: *mut i32,
        cap: c_int,
    ) -> c_int;
    pub fn keryx_infer_token_piece(model: *mut Model, token: i32, out: *mut c_char, cap: c_int) -> c_int;
    pub fn keryx_infer_is_eog(model: *mut Model, token: i32) -> c_int;

    pub fn keryx_infer_tensor_count(model: *mut Model) -> usize;
    pub fn keryx_infer_tensor_info(
        model: *mut Model,
        index: usize,
        name: *mut *const c_char,
        data: *mut *mut c_void,
        nbytes: *mut usize,
        is_device: *mut c_int,
    ) -> c_int;

    pub fn keryx_infer_vendor_products() -> u64;
    pub fn keryx_infer_free(model: *mut Model);
}
