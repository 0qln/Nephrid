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
macro_rules! impl_variant_utils_with_assertion {
    ($inner_type:ident as $type:ident) => {
        paste::paste! {
            impl $type {
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

#[macro_export]
macro_rules! mk_variants {
    ($inner_type:ident as $type:ident { $($name:ident = $value:expr),* $(,)? }) => {
        paste::paste! {
            $(
                pub const $name: $type = $type { v: $value };
                pub const [<$name _C>]: $inner_type = self::$name.v();
            )*
        }
    };
    ($inner_type:ident as $type:ident { $($name:ident),* $(,)? }) => {
        paste::paste! {
            mk_variants! { $inner_type as $type {
                $( $name = ${index()}, )*
            } }
        }
    };
    ($inner_type:ident as $type:ident { $($name:ident),* $(,)? } with_assert) => {
        paste::paste! {
            mk_variants! { $inner_type as $type { $( $name, )* } }
        }
        mk_variant_assert! { $inner_type as $type { $( $name, )* } }
    };
}

#[macro_export]
macro_rules! mk_variant_assert {
    ($inner_type:ident as $type:ident { $($name:ident),* $(,)? }) => {
        /// Assert that `v` is a valid variant.
        pub const fn debug_assert_variant(v: $inner_type) {
            debug_assert!(
                false $(|| v==$name.v)*,
                "v is not a valid variant."
            );
        }
    };
}

#[macro_export]
macro_rules! mk_variant_utils {
    // Case where values are specified
    ($inner_type:ident as $type:ident) => {
        impl $type {
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