//! Errors of the inference SDK.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The shared library could not be opened.
    Library(String),
    /// The library speaks another C or engine ABI.
    Abi { expected: u32, found: u32 },
    /// A required symbol is missing.
    Symbol(&'static str),
    /// The model could not be loaded.
    Model(String),
    /// The engine reported a failed call; the text is its own last_error.
    Call(String),
    /// The trace contract was not met before a call.
    Trace(String),
    /// The engine and the SDK disagree about the compatibility class.
    Class(String),
    /// A caller argument is invalid.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Library(e) => write!(f, "cannot open the inference library: {e}"),
            Self::Abi { expected, found } => {
                write!(f, "inference library ABI {found}, expected {expected}")
            }
            Self::Symbol(name) => write!(f, "inference library is missing {name}"),
            Self::Model(e) => write!(f, "model load refused: {e}"),
            Self::Call(e) => write!(f, "engine call failed: {e}"),
            Self::Trace(e) => write!(f, "trace contract: {e}"),
            Self::Class(e) => write!(f, "compatibility class: {e}"),
            Self::Invalid(e) => write!(f, "invalid argument: {e}"),
        }
    }
}

impl std::error::Error for Error {}
