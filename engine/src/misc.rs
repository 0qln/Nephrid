use std::{fmt::Debug, num::ParseIntError};

use thiserror::Error;

#[const_trait]
pub trait ConstFrom<T> {
    fn from_c(value: T) -> Self;
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Invalid value: {0}")]
    InputOutOfRange(String),

    #[error("Missing value")]
    MissingValue,

    #[error("ParseIntError: {0:?}")]
    ParseIntError(ParseIntError),
}

#[derive(Error, Debug)]
pub enum TokenizationError {
    #[error("Tokenizer exhaused")]
    TokenizerExhaused,
}

#[macro_export]
macro_rules! impl_variants_with_assertion {
    ($inner_type:ident as $type:ident { $($name:ident),* $(,)? }) => {
        paste::paste! {
            impl $type {
                $(
                    pub const $name: $type = $type { v: ${index()} };
                    pub const [<$name _C>]: $inner_type = Self::$name.v();
                )*
        
                /// The number of variants.
                pub const N_VARIANTS: usize = { 
                    0 $(+ { let _ = $type::$name; 1 })*
                };

                /// Get the value of the variant.
                #[inline]
                pub const fn v(&self) -> $inner_type {
                    self.v
                }

                /// Create a $type from a value that is a valid variant.
                /// This is unsafe, because the value is not checked.
                /// 
                /// # Safety
                /// Only use this if you are certain of v's range.
                #[inline]
                pub const unsafe fn from_v(v: $inner_type) -> Self {
                    $type { v }
                }

                /// Assert, that v is a variant of $type.
                #[inline]
                pub const fn assert_variant(v: $inner_type) {
                    debug_assert!(
                        false $(|| v == $type::$name.v)*,
                        "v is not a variant of type."
                    );
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_variants {
    // Case where values are specified
    ($inner_type:ident as $type:ident { $($name:ident = $value:expr),* $(,)? }) => {
        paste::paste! {
            impl $type {
                $(
                    pub const $name: $type = $type { v: $value };
                    pub const [<$name _C>]: $inner_type = Self::$name.v();
                )*
        
                /// The number of variants.
                pub const N_VARIANTS: usize = { 
                    0 $(+ { let _ = $type::$name; 1 })*
                };

                /// Get the value of the variant.
                #[inline]
                pub const fn v(&self) -> $inner_type {
                    self.v
                }

                /// Create a $type from a value that is a valid variant.
                /// This is unsafe, because the value is not checked.
                ///
                /// # Safety
                /// Only use this if you are certain of v's range.
                #[inline]
                pub const unsafe fn from_v(v: $inner_type) -> Self {
                    $type { v }
                }
            }
        }
    };
    // Case where values are not specified
    ($inner_type:ident as $type:ident { $($name:ident),* $(,)? }) => {
        paste::paste! {
            impl $type {
                $(
                    pub const $name: $type = $type { v: ${index()} };
                    pub const [<$name _C>]: $inner_type = Self::$name.v();
                )*
        
                /// The number of variants.
                pub const N_VARIANTS: usize = { 
                    0 $(+ { let _ = $type::$name; 1 })*
                };

                /// Get the value of the variant.
                #[inline]
                pub const fn v(&self) -> $inner_type {
                    self.v
                }

                /// Create a $type from a value that is a valid variant.
                /// This is unsafe, because the value is not checked.
                ///
                /// # Safety
                /// Only use this if you are certain of v's range.
                #[inline]
                pub const unsafe fn from_v(v: $inner_type) -> Self {
                    $type { v }
                }
            }
        }
    };
}

pub fn trim_newline(s: &mut String) {
    if s.ends_with('\n') {
        s.pop();
        if s.ends_with('\r') {
            s.pop();
        }
    }
}