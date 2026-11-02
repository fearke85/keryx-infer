//! `keryx-infer-conformance`: inspect the canonical manifest and compute compatibility ids.
//!
//! The native engine runner is added in the engine phase; this binary already lets two
//! implementations compare per-backend `patch_set` values and `compat_id`s without loading a
//! model.

use std::process::ExitCode;

use keryx_infer_protocol::{
    backend_patch_set, manifest, manifest_patch_set, Backend, CardClass, CompatClass, Record, UnitKind, Vendor, MAGIC,
};
use keryx_infer_vectors::VectorSet;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().map(String::as_str) else {
        usage();
        return ExitCode::FAILURE;
    };
    let rest = &args[1..];
    let result = match command {
        "manifest" => manifest_command(),
        "class-id" => class_id_command(rest),
        "inspect" => inspect_command(rest),
        "vectors" => vectors_command(),
        "help" | "-h" | "--help" => {
            usage();
            Ok(())
        }
        other => Err(format!("unknown command '{other}'")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!(
        "usage:\n  \
         keryx-infer-conformance manifest\n  \
         keryx-infer-conformance class-id --backend cuda --arch cc8.6 --vendor nvidia --units 84 [--unit-kind sm] [--require-ready]\n  \
         keryx-infer-conformance inspect <record>\n  \
         keryx-infer-conformance vectors"
    );
}

fn manifest_command() -> Result<(), String> {
    let m = manifest();
    println!("protocol: {}", m.protocol);
    println!("engine_abi: {}", m.engine_abi);
    println!("manifest_patch_set: {}", hex::encode(manifest_patch_set()));
    println!("trace_layout: {}", m.trace_layout);
    println!("prompt_template: {}", m.prompt_template);
    for (name, backend) in &m.backends {
        println!(
            "backend {name}: status={} planned_archs={} patch_set={}",
            backend.status,
            backend.plan.len(),
            backend_patch_set(name).map_or_else(|| "-".to_string(), hex::encode)
        );
        for (arch, row) in &backend.plan {
            println!("  {arch}: {}", row.units());
        }
    }
    Ok(())
}

fn class_id_command(args: &[String]) -> Result<(), String> {
    check_flags(args, &["--backend", "--arch", "--vendor", "--units", "--unit-kind"], &["--require-ready"])?;
    let backend = parse_flag(args, "--backend").ok_or("missing --backend")?;
    let arch = parse_flag(args, "--arch").ok_or("missing --arch")?;
    let vendor = parse_flag(args, "--vendor").ok_or("missing --vendor")?;
    let units =
        parse_flag(args, "--units").ok_or("missing --units")?.parse::<u16>().map_err(|e| format!("--units: {e}"))?;
    let backend = Backend::parse(&backend).ok_or_else(|| format!("unknown backend {backend}"))?;
    let vendor = Vendor::parse(&vendor).ok_or_else(|| format!("unknown vendor {vendor}"))?;
    let unit_kind = match parse_flag(args, "--unit-kind") {
        Some(value) => UnitKind::parse(&value).ok_or_else(|| format!("unknown unit kind {value}"))?,
        None => {
            manifest()
                .card_kind(backend.name(), &arch)
                .ok_or_else(|| format!("backend {}/{arch} has no unit kind", backend.name()))?
                .1
        }
    };
    let card = CardClass::new(backend, vendor, arch, units, unit_kind);
    let require_ready = args.iter().any(|arg| arg == "--require-ready");
    let class = CompatClass::from_card(manifest().engine_abi, card.clone(), require_ready)?;
    let status = manifest().backend(backend.name()).map_or("unknown", |backend| backend.status.as_str());
    println!("label: {}", card.label());
    println!("arch: {}", card.arch);
    println!("status: {status}");
    println!("planned_units: {}", card.planned_units);
    println!("unit_kind: {}", card.unit_kind().map_or("unknown", UnitKind::name));
    println!("patch_set: {}", hex::encode(class.patch_set));
    println!("compat_id: {}", hex::encode(class.id()));
    if status != "validated" {
        eprintln!("note: backend {status}; this class is not advertiseable yet");
    }
    Ok(())
}

fn inspect_command(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("missing record path")?;
    let blob = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let record = Record::decode(&blob)?;
    let class = record.class();
    println!("magic: {}", String::from_utf8_lossy(MAGIC));
    println!("protocol: {}", record.header.protocol);
    println!("engine_abi: {}", record.header.engine_abi);
    println!("compat_id: {}", hex::encode(record.header.compat_id));
    println!("model_id: {}", hex::encode(record.header.model_id));
    println!("tier: {}", record.header.tier);
    println!("window: {}", record.header.window);
    println!("cards: {}", record.header.cards.len());
    for card in &class.cards {
        println!(
            "  {} planned={} {}",
            card.label(),
            card.planned_units,
            card.unit_kind().map_or("unknown", UnitKind::name)
        );
    }
    println!("positions: {}", record.body.record.pos_roots.len());
    println!("prompt: {}", record.body.record.n_prompt);
    println!("generated: {}", record.generated_ids()?.len());
    Ok(())
}

fn vectors_command() -> Result<(), String> {
    let set = VectorSet::parse()?;
    println!("protocol={} engine_abi={} vectors={}", set.protocol, set.engine_abi, set.vectors.len());
    set.check()?;
    for vector in &set.vectors {
        println!("{}: patch_set={} compat_id={}", vector.name, vector.patch_set, vector.compat_id);
    }
    Ok(())
}

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|arg| arg == name).and_then(|index| args.get(index + 1)).cloned()
}

/// Refuses anything but the known flags, each at most once, so a removed, misspelled or repeated
/// flag is not silently ignored.
fn check_flags(args: &[String], with_value: &[&str], switches: &[&str]) -> Result<(), String> {
    let mut seen = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if seen.contains(&arg) {
            return Err(format!("{arg} is given more than once"));
        }
        seen.push(arg);
        if with_value.contains(&arg.as_str()) {
            match iter.next() {
                Some(value) if !value.starts_with("--") => {}
                _ => return Err(format!("{arg} needs a value")),
            }
        } else if !switches.contains(&arg.as_str()) {
            return Err(format!("unknown argument '{arg}'"));
        }
    }
    Ok(())
}
