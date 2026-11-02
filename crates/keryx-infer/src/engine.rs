//! Safe wrapper over one loaded, single-card inference engine.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

use keryx_infer_protocol::{CardClass, CompatClass, Hash};
use keryx_infer_sys as sys;

use crate::error::Error;
use crate::native::Native;
use crate::trace::{ReplayWants, Replayed, TraceDims, TraceSink, TracedRun};

/// A tensor of the canonical weight table, for a miner's zero-dup walk.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub data: *mut c_void,
    pub nbytes: usize,
    pub is_device: bool,
}

// The pointer is owned by the engine and valid while the engine is alive; the walk uses it under
// the same card lease as generation.
unsafe impl Send for Tensor {}

/// One engine on one card. The model is single-card by construction; the C ABI has no load_pair.
pub struct Engine {
    native: Arc<Native>,
    model: *mut sys::Model,
    class: CompatClass,
    n_ctx: i32,
}

// The raw model is only reached through `&mut self` or a shared read-only call; a miner wraps the
// engine in its own per-card lock.
unsafe impl Send for Engine {}

impl Engine {
    /// Opens the shared library and loads `gguf` on `gpu`, reading weights from `blob`.
    ///
    /// # Safety
    /// `blob` must point to at least `layout_len` readable bytes of the canonical walk blob and
    /// stay alive for the whole engine lifetime.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn load_bound(
        library: impl AsRef<Path>,
        gguf: &str,
        gpu: usize,
        blob: *mut c_void,
        held: usize,
        cut: usize,
        layout_len: usize,
        n_ctx: i32,
        card: CardClass,
    ) -> Result<Self, Error> {
        let gpu = c_int::try_from(gpu).map_err(|_| Error::Invalid("GPU ordinal past i32".into()))?;
        let class = CompatClass::single(sys::ENGINE_ABI, card.clone()).map_err(Error::Invalid)?;
        let native = Native::open(library.as_ref())?;

        let native_patch = native
            .patch_set(card.backend)
            .ok_or_else(|| Error::Class("the engine has no patch set for this backend".into()))?;
        if native_patch != class.patch_set {
            return Err(Error::Class(format!(
                "engine patch set {} differs from the SDK manifest {}",
                hex::encode(native_patch),
                hex::encode(class.patch_set)
            )));
        }
        let native_id = native.compat_id(&card)?;
        if native_id != class.id() {
            return Err(Error::Class(format!(
                "engine class {} differs from the SDK class {}",
                hex::encode(native_id),
                hex::encode(class.id())
            )));
        }
        // The class is only trustworthy if the device itself says it is this card; a caller-supplied
        // architecture cannot be checked against the plan table alone.
        let actual = native.card_for_device(card.backend, gpu).ok_or_else(|| {
            Error::Class(format!(
                "device {gpu} cannot report its class on backend {}; refusing to load an unverified class: {}",
                card.backend,
                native.last_error()
            ))
        })?;
        if actual != card {
            return Err(Error::Class(format!(
                "device reports {} planned {} {}, the class says {} planned {} {}",
                actual.label(),
                actual.planned_units,
                actual.unit_kind().map_or("unknown", keryx_infer_protocol::UnitKind::name),
                card.label(),
                card.planned_units,
                card.unit_kind().map_or("unknown", keryx_infer_protocol::UnitKind::name)
            )));
        }

        let gguf = CString::new(gguf).map_err(|_| Error::Invalid("GGUF path contains NUL".into()))?;
        let model = (native.load_bound)(gguf.as_ptr(), gpu, blob, held, cut, layout_len, n_ctx);
        if model.is_null() {
            return Err(Error::Model(native.last_error()));
        }
        Ok(Self { native, model, class, n_ctx })
    }

    pub fn card(&self) -> &CardClass {
        &self.class.cards[0]
    }

    pub fn n_ctx(&self) -> i32 {
        self.n_ctx
    }

    pub fn engine_abi(&self) -> u32 {
        self.class.engine_abi
    }

    pub fn version(&self) -> String {
        self.native.version()
    }

    /// The class id of the validated class. The native library is checked against it at load.
    pub fn compat_id(&self) -> Hash {
        self.class.id()
    }

    /// The class id computed by the native library, for diagnostics and conformance.
    pub fn native_compat_id(&self) -> Result<Hash, Error> {
        self.native.compat_id(self.card())
    }

    pub fn class(&self) -> &CompatClass {
        &self.class
    }

    pub fn patch_set(&self) -> Hash {
        self.class.patch_set
    }

    pub fn plan_units(&self, backend: u16, arch: &str, real_units: u16) -> Option<u16> {
        self.native.plan_units(backend, arch, real_units)
    }

    pub fn card_for_device(&self, backend: u16, device: i32) -> Option<CardClass> {
        self.native.card_for_device(backend, device)
    }

    pub fn vendor_products(&self) -> u64 {
        self.native.vendor_products()
    }

    /// Re-creates the context at `n_ctx` tokens; the weights stay in the blob.
    pub fn context_acquire(&mut self, n_ctx: i32) -> Result<i32, Error> {
        if n_ctx <= 0 {
            return Err(Error::Invalid("context size must be positive".into()));
        }
        let got = unsafe { (self.native.context_acquire)(self.model, n_ctx) };
        if got < 0 {
            return Err(Error::Call(self.native.last_error()));
        }
        self.n_ctx = got;
        Ok(got)
    }

    pub fn context_release(&mut self) {
        unsafe { (self.native.context_release)(self.model) };
        self.n_ctx = 0;
    }

    /// Arms the capture for a tier. The counts come from the protocol crate's tier context.
    #[allow(clippy::too_many_arguments)]
    pub fn trace_init(
        &mut self,
        manifest: &[u8],
        n_layers: u32,
        n_used: u32,
        n_attn: u32,
        n_recur: u32,
        seq_heads: u32,
    ) -> Result<TraceDims, Error> {
        let rc = unsafe {
            (self.native.trace_init)(
                self.model,
                manifest.as_ptr(),
                manifest.len(),
                n_layers,
                n_used,
                n_attn,
                n_recur,
                seq_heads,
            )
        };
        if rc != 0 {
            return Err(Error::Call(self.native.last_error()));
        }
        self.trace_dims()
    }

    pub fn trace_dims(&self) -> Result<TraceDims, Error> {
        let leaves = unsafe { (self.native.trace_leaves)(self.model) };
        let seq_leaves = unsafe { (self.native.trace_seq_leaves)(self.model) };
        if leaves == 0 || seq_leaves == 0 {
            return Err(Error::Trace("the engine has no armed capture".into()));
        }
        Ok(TraceDims { leaves, seq_leaves })
    }

    /// Runs `prompt` traced and hands every token and batch row to `sink`.
    pub fn generate_traced(
        &mut self,
        prompt: &[u32],
        max_tokens: u32,
        sink: &mut dyn TraceSink,
    ) -> Result<TracedRun, Error> {
        let dims = self.trace_dims()?;
        let prompt: Vec<i32> = prompt
            .iter()
            .map(|&t| i32::try_from(t).map_err(|_| Error::Invalid("token id past i32".into())))
            .collect::<Result<_, _>>()?;
        let n_prompt = c_int::try_from(prompt.len()).map_err(|_| Error::Invalid("prompt is too long".into()))?;
        let max_tokens = c_int::try_from(max_tokens).unwrap_or(c_int::MAX);
        let mut seq = vec![0u8; dims.seq_leaves as usize * 32];
        let mut call =
            Call { native: &self.native, sink, model: self.model, leaves: dims.leaves as usize, failed: None };
        let n = unsafe {
            (self.native.generate_traced)(
                self.model,
                prompt.as_ptr(),
                n_prompt,
                max_tokens,
                on_token,
                on_rows,
                &mut call as *mut Call as *mut c_void,
                seq.as_mut_ptr(),
                seq.len(),
            )
        };
        if let Some(why) = call.failed {
            return Err(Error::Trace(format!("generation stopped: {why}")));
        }
        if n < 0 {
            return Err(Error::Call(self.native.last_error()));
        }
        Ok(TracedRun { generated: n as u32, seq_hashes: seq.as_chunks::<32>().0.to_vec() })
    }

    /// Replays a kept run and copies the leaves `wants` names. Every batch's leaf table goes to
    /// `rows`; the caller checks it against the kept roots before signing anything.
    pub fn replay(
        &mut self,
        tokens: &[u32],
        n_prompt: u32,
        wants: &ReplayWants,
        rows: &mut dyn FnMut(u32, u32, &[u8]) -> bool,
    ) -> Result<Replayed, Error> {
        let dims = self.trace_dims()?;
        let tokens: Vec<i32> = tokens
            .iter()
            .map(|&t| i32::try_from(t).map_err(|_| Error::Invalid("token id past i32".into())))
            .collect::<Result<_, _>>()?;
        let n_pos = c_int::try_from(tokens.len()).map_err(|_| Error::Invalid("run is too long".into()))?;
        let n_prompt = c_int::try_from(n_prompt).map_err(|_| Error::Invalid("prompt is too long".into()))?;
        if n_prompt <= 0 || n_prompt > n_pos {
            return Err(Error::Invalid("prompt outside the run".into()));
        }
        let mut pos_wants = Vec::with_capacity(wants.pos.len() * 3);
        for &(t, leaf, offset) in &wants.pos {
            pos_wants.extend([to_i64(t)?, to_i64(leaf)?, to_i64(offset)?]);
        }
        let mut seq_wants = Vec::with_capacity(wants.seq.len() * 2);
        for &(leaf, offset) in &wants.seq {
            seq_wants.extend([to_i64(leaf)?, to_i64(offset)?]);
        }
        let n_pos_wants =
            c_int::try_from(wants.pos.len()).map_err(|_| Error::Invalid("too many position wants".into()))?;
        let n_seq_wants =
            c_int::try_from(wants.seq.len()).map_err(|_| Error::Invalid("too many sequence wants".into()))?;
        let mut pos_out = vec![0u8; wants.pos_bytes];
        let mut seq_out = vec![0u8; wants.seq_bytes];
        let mut seq_hashes = vec![0u8; dims.seq_leaves as usize * 32];
        let mut sink = RowsOnly(rows);
        let mut call = Call {
            native: &self.native,
            sink: &mut sink,
            model: self.model,
            leaves: dims.leaves as usize,
            failed: None,
        };
        let rc = unsafe {
            (self.native.replay)(
                self.model,
                tokens.as_ptr(),
                n_pos,
                n_prompt,
                pos_wants.as_ptr(),
                n_pos_wants,
                pos_out.as_mut_ptr(),
                pos_out.len(),
                seq_wants.as_ptr(),
                n_seq_wants,
                seq_out.as_mut_ptr(),
                seq_out.len(),
                on_rows,
                &mut call as *mut Call as *mut c_void,
                seq_hashes.as_mut_ptr(),
                seq_hashes.len(),
            )
        };
        if let Some(why) = call.failed {
            return Err(Error::Trace(format!("replay stopped: {why}")));
        }
        if rc != 0 {
            return Err(Error::Call(self.native.last_error()));
        }
        Ok(Replayed { pos: pos_out, seq: seq_out, seq_hashes: seq_hashes.as_chunks::<32>().0.to_vec() })
    }

    /// The model's token ids for `text`. `add_special` adds BOS; `parse_special` parses control
    /// tokens. The engine owns tokenization so every miner builds the same prompt stream.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<u32>, Error> {
        let len = c_int::try_from(text.len()).map_err(|_| Error::Invalid("text is too long".into()))?;
        let mut cap = text.len() + 16;
        for _ in 0..2 {
            let mut out = vec![0i32; cap];
            let n = unsafe {
                (self.native.tokenize)(
                    self.model,
                    text.as_ptr() as *const c_char,
                    len,
                    c_int::from(add_special),
                    c_int::from(parse_special),
                    out.as_mut_ptr(),
                    cap as c_int,
                )
            };
            if n >= 0 {
                out.truncate(n as usize);
                return out
                    .into_iter()
                    .map(|t| u32::try_from(t).map_err(|_| Error::Call("negative token id".into())))
                    .collect();
            }
            cap = n.unsigned_abs() as usize;
        }
        Err(Error::Call(self.native.last_error()))
    }

    pub fn token_piece(&self, token: u32) -> Result<Vec<u8>, Error> {
        let token = i32::try_from(token).map_err(|_| Error::Invalid("token id past i32".into()))?;
        let mut buf = [0u8; 256];
        let n = unsafe {
            (self.native.token_piece)(self.model, token, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int)
        };
        if n < 0 || n as usize > buf.len() {
            return Err(Error::Call(self.native.last_error()));
        }
        Ok(buf[..n as usize].to_vec())
    }

    pub fn is_eog(&self, token: u32) -> bool {
        i32::try_from(token).is_ok_and(|t| unsafe { (self.native.is_eog)(self.model, t) != 0 })
    }

    /// The canonical weight table, for a zero-dup walk over the engine's bytes.
    pub fn tensors(&self) -> Result<Vec<Tensor>, Error> {
        let n = unsafe { (self.native.tensor_count)(self.model) };
        let mut out = Vec::with_capacity(n);
        for index in 0..n {
            let mut name: *const c_char = std::ptr::null();
            let mut data: *mut c_void = std::ptr::null_mut();
            let mut nbytes = 0usize;
            let mut is_device = 0i32;
            let ok = unsafe {
                (self.native.tensor_info)(self.model, index, &mut name, &mut data, &mut nbytes, &mut is_device)
            };
            if ok == 0 || name.is_null() || data.is_null() {
                return Err(Error::Call(self.native.last_error()));
            }
            out.push(Tensor {
                name: unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned(),
                data,
                nbytes,
                is_device: is_device != 0,
            });
        }
        Ok(out)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.model.is_null() {
            unsafe { (self.native.free)(self.model) };
            self.model = std::ptr::null_mut();
        }
    }
}

fn to_i64(value: u64) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::Invalid("replay want past i64".into()))
}

struct Call<'a> {
    native: &'a Native,
    sink: &'a mut dyn TraceSink,
    model: *mut sys::Model,
    leaves: usize,
    failed: Option<&'static str>,
}

struct RowsOnly<'a>(&'a mut dyn FnMut(u32, u32, &[u8]) -> bool);

impl TraceSink for RowsOnly<'_> {
    fn token(&mut self, _token: u32, _piece: &[u8]) -> bool {
        false
    }

    fn rows(&mut self, t0: u32, n_ubatch: u32, hashes: &[u8]) -> bool {
        (self.0)(t0, n_ubatch, hashes)
    }
}

extern "C" fn on_token(user: *mut c_void, token: i32) -> c_int {
    let call = unsafe { &mut *(user as *mut Call) };
    let fed = catch_unwind(AssertUnwindSafe(|| {
        let Ok(id) = u32::try_from(token) else {
            call.failed = Some("negative token id");
            return false;
        };
        let mut buf = [0u8; 256];
        let n = unsafe {
            (call.native.token_piece)(call.model, token, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int)
        };
        if n < 0 || n as usize > buf.len() {
            call.failed = Some("token text unavailable");
            return false;
        }
        call.sink.token(id, &buf[..n as usize])
    }));
    c_int::from(fed.unwrap_or_else(|_| {
        call.failed = Some("token callback panicked");
        false
    }))
}

extern "C" fn on_rows(user: *mut c_void, t0: i32, n_rows: i32, n_ubatch: i32, hashes: *const u8) -> c_int {
    let call = unsafe { &mut *(user as *mut Call) };
    let ok = catch_unwind(AssertUnwindSafe(|| {
        let (Ok(t0), Ok(rows), Ok(n_ubatch)) = (u32::try_from(t0), usize::try_from(n_rows), u32::try_from(n_ubatch))
        else {
            call.failed = Some("negative row bounds");
            return false;
        };
        if hashes.is_null() {
            call.failed = Some("no hash table");
            return false;
        }
        let Some(len) = rows.checked_mul(call.leaves).and_then(|n| n.checked_mul(32)) else {
            call.failed = Some("hash table overflow");
            return false;
        };
        let table = unsafe { std::slice::from_raw_parts(hashes, len) };
        call.sink.rows(t0, n_ubatch, table)
    }));
    c_int::from(ok.unwrap_or_else(|_| {
        call.failed = Some("rows callback panicked");
        false
    }))
}
