use std::{any::Any, num::ParseIntError, ops::AddAssign};

use thiserror::Error;

pub trait PostIncrement<T> {
    fn post_incr(&mut self, incr: T) -> T;
}

impl<T> PostIncrement<T> for T
where
    T: AddAssign + Copy,
{
    #[inline]
    fn post_incr(&mut self, incr: T) -> T {
        let tmp = *self;
        *self += incr;
        tmp
    }
}

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

                /// Get the value of the variant.
                #[inline]
                pub const fn v(&self) -> $inner_type {
                    self.v
                }

                /// Create a $type from a value that is a valid variant.
                /// This is unsafe, because the value is not checked.
                /// Only use this if you are certain of v's range.
                #[inline]
                pub const unsafe fn from_v(v: $inner_type) -> Self {
                    $type { v }
                }

                /// Assert, that v is a variant of $type.
                #[inline]
                pub const fn assert_variant(v: $inner_type) {
                    assert!(
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

                /// Get the value of the variant.
                #[inline]
                pub const fn v(&self) -> $inner_type {
                    self.v
                }

                /// Create a $type from a value that is a valid variant.
                /// This is unsafe, because the value is not checked.
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

                /// Get the value of the variant.
                #[inline]
                pub const fn v(&self) -> $inner_type {
                    self.v
                }

                /// Create a $type from a value that is a valid variant.
                /// This is unsafe, because the value is not checked.
                /// Only use this if you are certain of v's range.
                #[inline]
                pub const unsafe fn from_v(v: $inner_type) -> Self {
                    $type { v }
                }
            }
        }
    };
}
