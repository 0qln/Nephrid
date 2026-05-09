pub mod tokens;

use std::fmt::Debug;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UciError {
    #[error("Expected value for {0}")]
    MissingArgument(&'static str),

    #[error("Value {0} is out of range: [{1}; {2}]")]
    InputOutOfRange(String, String, String),

    #[error("Invalid value {0}, expected one of: {1:?}")]
    InvalidValue(String, Vec<String>),

    #[error("Invalid command: {0}")]
    InvalidCommand(String),

    #[error("Unknown option: {0}")]
    UnknownOption(String),
}
