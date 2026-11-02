//! Dynamic loading and the raw function table.
//!
//! The library is leaked for the process lifetime, so every resolved function pointer stays
//! valid. This mirrors how the miners load their sidecar today, without linking the engine into
//! the miner binary.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use keryx_infer_protocol::{CardClass, Hash};
use keryx_infer_sys as sys;

use crate::error::Error;

type CAbiFn = unsafe extern "C" fn() -> c_int;
type EngineAbiFn = unsafe extern "C" fn() -> c_int;
type VersionFn = unsafe extern "C" fn() -> *const c_char;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
type PlanUnitsFn = unsafe extern "C" fn(u16, *const c_char, c_int) -> c_int;
type CardForDeviceFn = unsafe extern "C" fn(u16, c_int, *mut sys::CardClass) -> c_int;
type PatchSetFn = unsafe extern "C" fn(u16, *mut u8) -> c_int;
type CompatIdFn = unsafe extern "C" fn(*const sys::CardClass, usize, *mut u8) -> c_int;
type LoadBoundFn =
    unsafe extern "C" fn(*const c_char, c_int, *mut c_void, usize, usize, usize, c_int) -> *mut sys::Model;
type ContextAcquireFn = unsafe extern "C" fn(*mut sys::Model, c_int) -> c_int;
type ContextReleaseFn = unsafe extern "C" fn(*mut sys::Model);
type TraceInitFn = unsafe extern "C" fn(*mut sys::Model, *const u8, usize, u32, u32, u32, u32, u32) -> c_int;
type TraceLeavesFn = unsafe extern "C" fn(*mut sys::Model) -> u32;
type TraceSeqLeavesFn = unsafe extern "C" fn(*mut sys::Model) -> u32;
type GenerateTracedFn = unsafe extern "C" fn(
    *mut sys::Model,
    *const i32,
    c_int,
    c_int,
    sys::OnToken,
    sys::OnRows,
    *mut c_void,
    *mut u8,
    usize,
) -> c_int;
type ReplayFn = unsafe extern "C" fn(
    *mut sys::Model,
    *const i32,
    c_int,
    c_int,
    *const i64,
    c_int,
    *mut u8,
    usize,
    *const i64,
    c_int,
    *mut u8,
    usize,
    sys::OnRows,
    *mut c_void,
    *mut u8,
    usize,
) -> c_int;
type TokenizeFn = unsafe extern "C" fn(*mut sys::Model, *const c_char, c_int, c_int, c_int, *mut i32, c_int) -> c_int;
type TokenPieceFn = unsafe extern "C" fn(*mut sys::Model, i32, *mut c_char, c_int) -> c_int;
type IsEogFn = unsafe extern "C" fn(*mut sys::Model, i32) -> c_int;
type TensorCountFn = unsafe extern "C" fn(*mut sys::Model) -> usize;
type TensorInfoFn =
    unsafe extern "C" fn(*mut sys::Model, usize, *mut *const c_char, *mut *mut c_void, *mut usize, *mut c_int) -> c_int;
type VendorProductsFn = unsafe extern "C" fn() -> u64;
type FreeFn = unsafe extern "C" fn(*mut sys::Model);

/// Resolved C ABI of one loaded engine library.
pub(crate) struct Native {
    pub(crate) c_abi: CAbiFn,
    pub(crate) engine_abi: EngineAbiFn,
    pub(crate) version: VersionFn,
    pub(crate) last_error: LastErrorFn,
    pub(crate) plan_units: PlanUnitsFn,
    pub(crate) card_for_device: CardForDeviceFn,
    pub(crate) patch_set: PatchSetFn,
    pub(crate) compat_id: CompatIdFn,
    pub(crate) load_bound: LoadBoundFn,
    pub(crate) context_acquire: ContextAcquireFn,
    pub(crate) context_release: ContextReleaseFn,
    pub(crate) trace_init: TraceInitFn,
    pub(crate) trace_leaves: TraceLeavesFn,
    pub(crate) trace_seq_leaves: TraceSeqLeavesFn,
    pub(crate) generate_traced: GenerateTracedFn,
    pub(crate) replay: ReplayFn,
    pub(crate) tokenize: TokenizeFn,
    pub(crate) token_piece: TokenPieceFn,
    pub(crate) is_eog: IsEogFn,
    pub(crate) tensor_count: TensorCountFn,
    pub(crate) tensor_info: TensorInfoFn,
    pub(crate) vendor_products: VendorProductsFn,
    pub(crate) free: FreeFn,
}

impl Native {
    /// Opens `path` once per process and checks the C and engine ABIs before returning it.
    ///
    /// The cache is keyed by path and checked against the file's content hash. A touched or
    /// re-copied identical file reuses the loaded library. A file whose bytes changed is refused:
    /// a loaded library is never unloaded, and whether the dynamic loader maps a second copy or
    /// hands back the first depends on how the file was replaced, so the only reliable upgrade is
    /// a restart.
    pub(crate) fn open(path: &Path) -> Result<Arc<Self>, Error> {
        let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let identity = content_hash(&key)?;
        let mut cache = cache().lock().unwrap_or_else(|p| p.into_inner());
        if let Some((loaded, native)) = cache.get(&key) {
            if *loaded != identity {
                return Err(Error::Library(format!(
                    "{}: the library was replaced after it was loaded; restart the process to use the new build",
                    key.display()
                )));
            }
            return Ok(Arc::clone(native));
        }
        let native = Self::load(path)?;
        cache.insert(key, (identity, Arc::clone(&native)));
        Ok(native)
    }

    fn load(path: &Path) -> Result<Arc<Self>, Error> {
        let library = unsafe { libloading::Library::new(path) }
            .map_err(|e| Error::Library(format!("{}: {e}", path.display())))?;

        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                unsafe { library.get::<$ty>($name).map(|symbol| *symbol).map_err(|_| Error::Symbol(name_str($name)))? }
            }};
        }

        let native = Self {
            c_abi: sym!(b"keryx_infer_c_abi\0", CAbiFn),
            engine_abi: sym!(b"keryx_infer_engine_abi\0", EngineAbiFn),
            version: sym!(b"keryx_infer_version\0", VersionFn),
            last_error: sym!(b"keryx_infer_last_error\0", LastErrorFn),
            plan_units: sym!(b"keryx_infer_plan_units\0", PlanUnitsFn),
            card_for_device: sym!(b"keryx_infer_card_for_device\0", CardForDeviceFn),
            patch_set: sym!(b"keryx_infer_manifest_patch_set\0", PatchSetFn),
            compat_id: sym!(b"keryx_infer_compat_id\0", CompatIdFn),
            load_bound: sym!(b"keryx_infer_load_bound\0", LoadBoundFn),
            context_acquire: sym!(b"keryx_infer_context_acquire\0", ContextAcquireFn),
            context_release: sym!(b"keryx_infer_context_release\0", ContextReleaseFn),
            trace_init: sym!(b"keryx_infer_trace_init\0", TraceInitFn),
            trace_leaves: sym!(b"keryx_infer_trace_leaves\0", TraceLeavesFn),
            trace_seq_leaves: sym!(b"keryx_infer_trace_seq_leaves\0", TraceSeqLeavesFn),
            generate_traced: sym!(b"keryx_infer_generate_traced\0", GenerateTracedFn),
            replay: sym!(b"keryx_infer_replay\0", ReplayFn),
            tokenize: sym!(b"keryx_infer_tokenize\0", TokenizeFn),
            token_piece: sym!(b"keryx_infer_token_piece\0", TokenPieceFn),
            is_eog: sym!(b"keryx_infer_is_eog\0", IsEogFn),
            tensor_count: sym!(b"keryx_infer_tensor_count\0", TensorCountFn),
            tensor_info: sym!(b"keryx_infer_tensor_info\0", TensorInfoFn),
            vendor_products: sym!(b"keryx_infer_vendor_products\0", VendorProductsFn),
            free: sym!(b"keryx_infer_free\0", FreeFn),
        };

        let c_abi = unsafe { (native.c_abi)() };
        if c_abi != sys::C_ABI as c_int {
            return Err(Error::Abi { expected: sys::C_ABI, found: c_abi.max(0) as u32 });
        }
        let engine_abi = unsafe { (native.engine_abi)() };
        if engine_abi != sys::ENGINE_ABI as c_int {
            return Err(Error::Abi { expected: sys::ENGINE_ABI, found: engine_abi.max(0) as u32 });
        }
        // Keep the library for the process lifetime only after the handshake succeeded. The
        // function table resolved above stays valid because this handle is never closed.
        let _: &'static libloading::Library = Box::leak(Box::new(library));
        Ok(Arc::new(native))
    }

    pub(crate) fn last_error(&self) -> String {
        let ptr = unsafe { (self.last_error)() };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        }
    }

    pub(crate) fn version(&self) -> String {
        let ptr = unsafe { (self.version)() };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        }
    }

    pub(crate) fn plan_units(&self, backend: u16, arch: &str, real_units: u16) -> Option<u16> {
        let arch = CString::new(arch).ok()?;
        let n = unsafe { (self.plan_units)(backend, arch.as_ptr(), c_int::from(real_units)) };
        u16::try_from(n).ok().filter(|n| *n > 0)
    }

    pub(crate) fn card_for_device(&self, backend: u16, device: i32) -> Option<CardClass> {
        let mut out = sys::CardClass::default();
        let rc = unsafe { (self.card_for_device)(backend, device, &mut out) };
        if rc != 0 {
            return None;
        }
        card_from_sys(&out)
    }

    pub(crate) fn patch_set(&self, backend: u16) -> Option<Hash> {
        let mut out = [0u8; 32];
        let rc = unsafe { (self.patch_set)(backend, out.as_mut_ptr()) };
        (rc == 0).then_some(out)
    }

    pub(crate) fn compat_id(&self, card: &CardClass) -> Result<Hash, Error> {
        let card = card_to_sys(card)?;
        let mut out = [0u8; 32];
        let rc = unsafe { (self.compat_id)(&card, 1, out.as_mut_ptr()) };
        if rc != 0 {
            return Err(Error::Call(self.last_error()));
        }
        Ok(out)
    }

    pub(crate) fn vendor_products(&self) -> u64 {
        unsafe { (self.vendor_products)() }
    }
}

type Cache = Mutex<HashMap<PathBuf, (Hash, Arc<Native>)>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// BLAKE3 of the library file, so the cache follows the bytes and not the file's metadata.
fn content_hash(path: &Path) -> Result<Hash, Error> {
    let file = std::fs::File::open(path).map_err(|e| Error::Library(format!("{}: {e}", path.display())))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(file).map_err(|e| Error::Library(format!("{}: {e}", path.display())))?;
    Ok(*hasher.finalize().as_bytes())
}

fn name_str(name: &[u8]) -> &'static str {
    // The literals above are NUL-terminated ASCII; strip the NUL at compile time is not trivial
    // in a macro, so this is used only for diagnostics.
    match name {
        b"keryx_infer_c_abi\0" => "keryx_infer_c_abi",
        b"keryx_infer_engine_abi\0" => "keryx_infer_engine_abi",
        b"keryx_infer_version\0" => "keryx_infer_version",
        b"keryx_infer_last_error\0" => "keryx_infer_last_error",
        b"keryx_infer_plan_units\0" => "keryx_infer_plan_units",
        b"keryx_infer_card_for_device\0" => "keryx_infer_card_for_device",
        b"keryx_infer_manifest_patch_set\0" => "keryx_infer_manifest_patch_set",
        b"keryx_infer_compat_id\0" => "keryx_infer_compat_id",
        b"keryx_infer_load_bound\0" => "keryx_infer_load_bound",
        b"keryx_infer_context_acquire\0" => "keryx_infer_context_acquire",
        b"keryx_infer_context_release\0" => "keryx_infer_context_release",
        b"keryx_infer_trace_init\0" => "keryx_infer_trace_init",
        b"keryx_infer_trace_leaves\0" => "keryx_infer_trace_leaves",
        b"keryx_infer_trace_seq_leaves\0" => "keryx_infer_trace_seq_leaves",
        b"keryx_infer_generate_traced\0" => "keryx_infer_generate_traced",
        b"keryx_infer_replay\0" => "keryx_infer_replay",
        b"keryx_infer_tokenize\0" => "keryx_infer_tokenize",
        b"keryx_infer_token_piece\0" => "keryx_infer_token_piece",
        b"keryx_infer_is_eog\0" => "keryx_infer_is_eog",
        b"keryx_infer_tensor_count\0" => "keryx_infer_tensor_count",
        b"keryx_infer_tensor_info\0" => "keryx_infer_tensor_info",
        b"keryx_infer_vendor_products\0" => "keryx_infer_vendor_products",
        b"keryx_infer_free\0" => "keryx_infer_free",
        _ => "unknown symbol",
    }
}

fn card_from_sys(card: &sys::CardClass) -> Option<CardClass> {
    let arch = c_arch(&card.arch)?;
    Some(CardClass {
        backend: card.backend,
        vendor: card.vendor,
        arch,
        planned_units: card.planned_units,
        unit_kind: card.unit_kind,
    })
}

fn card_to_sys(card: &CardClass) -> Result<sys::CardClass, Error> {
    let bytes = card.arch.as_bytes();
    if bytes.len() >= sys::ARCH_MAX {
        return Err(Error::Invalid("architecture id is too long for the C ABI".into()));
    }
    let mut arch = [0 as c_char; sys::ARCH_MAX];
    for (dst, src) in arch.iter_mut().zip(bytes) {
        *dst = *src as c_char;
    }
    Ok(sys::CardClass {
        backend: card.backend,
        vendor: card.vendor,
        arch,
        planned_units: card.planned_units,
        unit_kind: card.unit_kind,
    })
}

fn c_arch(arch: &[c_char; sys::ARCH_MAX]) -> Option<String> {
    let end = arch.iter().position(|&c| c == 0).unwrap_or(arch.len());
    let bytes: Vec<u8> = arch[..end].iter().map(|&c| c as u8).collect();
    String::from_utf8(bytes).ok().filter(|s| !s.is_empty())
}
