use std::{any::Any, num::ParseIntError};

use thiserror::Error;

#[const_trait]
pub trait ConstFrom<T> {
    fn from_c(value: T) -> Self;
}

#[const_trait]
pub trait ConstDefault {
    fn default_c() -> Self;
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Invalid value: {0:?}")]
    InputOutOfRange(Box<dyn Any>),
    
    #[error("Missing value")]
    MissingInput,
    
    #[error("ParseIntError: {0:?}")]
    ParseIntError(ParseIntError)
}

#[derive(Error, Debug)]
pub enum TokenizationError {
    #[error("Tokenizer exhaused")]
    TokenizerExhaused,
}


#[macro_export]
macro_rules! impl_variants {
    // Case where values are specified
    ($type:ident { $($name:ident = $value:expr),* $(,)? }) => {
        impl $type {
            $(
                pub const $name: $type = $type { v: $value };
            )*
        }
    };
    // Case where values are not specified
    ($type:ident { $($name:ident),* $(,)? }) => {
        impl $type {
            $(
                pub const $name: $type = $type { v: ${index()} };
            )*
        }
    };
}