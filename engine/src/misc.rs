use std::{
    any::type_name,
    fmt::{self, Debug},
    marker::PhantomData,
    ops::{Bound, IntoBounds},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use thiserror::Error;

pub const trait ConstFrom<T> {
    fn from_c(value: T) -> Self;
}

#[derive(Debug)]
pub struct Bounds<T, D>(Bound<T>, Bound<T>, PhantomData<D>);

impl<T, D> Bounds<T, D> {
    pub fn new<B: IntoBounds<T>>(bounds: B) -> Self {
        let (begin, end) = bounds.into_bounds();
        Self(begin, end, PhantomData)
    }
}

#[derive(Debug)]
pub struct DisplayFmt;

#[derive(Debug)]
pub struct DisplayDebug;

impl<T: fmt::Display> fmt::Display for Bounds<T, DisplayFmt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Bound::Included(x) => write!(f, "{x}=")?,
            Bound::Excluded(x) => write!(f, "{x}")?,
            Bound::Unbounded => {}
        };
        write!(f, "..")?;
        match &self.1 {
            Bound::Included(x) => write!(f, "={x}")?,
            Bound::Excluded(x) => write!(f, "{x}")?,
            Bound::Unbounded => {}
        };
        Ok(())
    }
}

impl<T: Debug> fmt::Display for Bounds<T, DisplayDebug> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Bound::Included(x) => write!(f, "{x:?}=")?,
            Bound::Excluded(x) => write!(f, "{x:?}")?,
            Bound::Unbounded => {}
        };
        write!(f, "..")?;
        match &self.1 {
            Bound::Included(x) => write!(f, "={x:?}")?,
            Bound::Excluded(x) => write!(f, "{x:?}")?,
            Bound::Unbounded => {}
        };
        Ok(())
    }
}

#[derive(Debug, Error)]
#[error("{value} was not in the range {range:?}")]
pub struct ValueOutOfRangeError<T: Debug> {
    pub value: T,
    pub range: Bounds<T, DisplayDebug>,
}

impl<T: Debug> ValueOutOfRangeError<T> {
    pub fn new<B: IntoBounds<T>>(value: T, range: B) -> Self {
        Self { value, range: Bounds::new(range) }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{value} was not in the set of expected values {expected:?}")]
pub struct ValueOutOfSetError<T: Debug + 'static> {
    pub value: T,
    pub expected: &'static [T],
}

impl<T: Debug + 'static> ValueOutOfSetError<T> {
    pub fn new(value: T, set: &'static [T]) -> Self {
        Self { value, expected: set }
    }
}

#[derive(Debug, Error)]
#[error("The specified value {value} was not a valid value.")]
pub struct InvalidValueError<T: Debug> {
    pub value: T,
}

impl<T: Debug> InvalidValueError<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

#[derive(Debug, Error, Default)]
#[error("Unexpected end of token stream. Expected a {} instead.", type_name::<T>())]
pub struct MissingTokenError<T: Default> {
    _t: PhantomData<T>,
}

impl<T: Default> MissingTokenError<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Error)]
#[error("Expected a {expected_token_name}, but got <something else>.")]
pub struct UnexpectedTokenError {
    pub expected_token_name: &'static str,
}

impl UnexpectedTokenError {
    pub fn new(token_name: &'static str) -> Self {
        Self { expected_token_name: token_name }
    }
}

// #[derive(Debug, Error)]
// #[error("The specified value was not in the correct format for a {}:
// {source}", type_name::<T>())] pub struct ParseTokenError<T, E: Error> {
//     pub error: E,
//     pub _t: PhantomData<T>,
// }

// impl<T, S: Error> ParseTokenError<T, S> {
//     pub fn new(source: S) -> Self {
//         Self { _t: PhantomData, source }
//     }
// }

#[macro_export]
macro_rules! impl_variants_with_assertion {
    ($inner_type:ident as $type:ident in $mod:ident { $($name:ident),* $(,)? }) => {
        paste::paste! {
            pub mod $mod {
                use super::*;
                $(
                    pub const $name: $type = $type { v: ${index()} };
                    pub const [<$name _C>]: $inner_type = self::$name.v();
                )*
            }

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

                /// Assert, that v is a variant of $type.
                #[inline]
                pub const fn assert_variant(v: $inner_type) {
                    debug_assert!(
                        false $(|| v == $mod::$name.v)*,
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
    ($inner_type:ident as $type:ident in $mod:ident { $($name:ident = $value:expr),* $(,)? }) => {
        paste::paste! {
            pub mod $mod {
                use super::*;
                $(
                    pub const $name: $type = $type { v: $value };
                    pub const [<$name _C>]: $inner_type = self::$name.v();
                )*
            }

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
    // Case where values are not specified
    ($inner_type:ident as $type:ident in $mod:ident { $($name:ident),* $(,)? }) => {
        paste::paste! {
             pub mod $mod {
                use super::*;
                $(
                    pub const $name: $type = $type { v: ${index()} };
                    pub const [<$name _C>]: $inner_type = self::$name.v();
                )*
            }

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

pub fn trim_newline(s: &mut String) {
    if s.ends_with('\n') {
        s.pop();
        if s.ends_with('\r') {
            s.pop();
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct DebugMode(Arc<AtomicBool>);

impl DebugMode {
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn set(&self, val: bool) {
        self.0.store(val, Ordering::Relaxed);
    }
}
