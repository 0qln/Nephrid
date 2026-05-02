use core::slice;
use std::{
    any::type_name,
    fmt::{self, Debug},
    iter,
    marker::PhantomData,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Bound, IntoBounds},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use thiserror::Error;

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

#[derive(Debug, Error, PartialEq)]
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

                /// The number of variants.
                pub const N_VARIANTS: usize = {
                    0 $(+ { let _ = self::$name; 1 })*
                };

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

                /// The number of variants.
                pub const N_VARIANTS: usize = {
                    0 $(+ { let _ = self::$name; 1 })*
                };
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

                /// The number of variants.
                pub const N_VARIANTS: usize = {
                    0 $(+ { let _ = self::$name; 1 })*
                };
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
    pub fn off() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn on() -> Self {
        Self(Arc::new(AtomicBool::new(true)))
    }

    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn set(&self, val: bool) {
        self.0.store(val, Ordering::Relaxed);
    }
}

#[derive(Default, Clone, Debug)]
pub struct CancellationToken {
    v: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            v: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.v.store(true, Ordering::Relaxed)
    }

    pub fn is_cancelled(&self) -> bool {
        self.v.load(Ordering::Relaxed)
    }
}

pub type CheckHealthResult<E> = Result<(), E>;

pub trait CheckHealth {
    type Error;

    #[must_use]
    fn check_health(&self) -> CheckHealthResult<Self::Error>;
}

#[repr(C)]
pub struct List<const N: usize, T> {
    items: [MaybeUninit<T>; N],
    len: usize,
}

impl<const N: usize, T: PartialEq> PartialEq for List<N, T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<const N: usize, T: Clone> Clone for List<N, T> {
    fn clone(&self) -> Self {
        let mut new_list = Self::new();
        new_list.len = self.len;
        // todo: use memcopy?
        for item in self.as_slice() {
            unsafe {
                new_list
                    .items
                    .get_unchecked_mut(self.len)
                    .write(item.clone());
            }
        }
        new_list
    }
}

impl<const N: usize, T> List<N, T> {
    /// Creates a new, empty list without initializing the underlying array.
    #[inline]
    pub const fn new() -> Self {
        Self {
            // SAFETY: An uninitialized array of `MaybeUninit` requires no initialization
            // and is perfectly valid to assume initialized.
            items: unsafe { MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Pushes an item to the list.
    #[inline]
    pub fn push(&mut self, item: T) {
        debug_assert!(self.len < N, "List capacity exceeded");
        // SAFETY: The bounds check is handled by the caller or by design constraints
        // (like the maximum 218 legal chess moves).
        unsafe {
            self.items.get_unchecked_mut(self.len).write(item);
        }
        self.len += 1;
    }

    /// Returns a mutable slice of the initialized elements.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: We only cast the slice up to `self.len`, which we
        // guarantee has been initialized via the `push` method.
        unsafe { slice::from_raw_parts_mut(self.items.as_mut_ptr().cast(), self.len) }
    }

    /// Returns a slice of the initialized elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: We only cast the slice up to `self.len`, which we
        // guarantee has been initialized via the `push` method.
        unsafe { slice::from_raw_parts(self.items.as_ptr().cast(), self.len) }
    }

    /// Returns an iterator over the initialized elements.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the initialized elements.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// # Safety
    /// The caller must ensure that `T` and `T2` have the exact same size and
    /// alignment, and that the bitwise representation of `T2` is a valid
    /// representation of `T`.
    pub unsafe fn transmute<T2>(src: List<N, T2>) -> List<N, T> {
        debug_assert_eq!(
            mem::size_of::<T>(),
            mem::size_of::<T2>(),
            "Sizes must match"
        );
        debug_assert_eq!(
            mem::align_of::<T>(),
            mem::align_of::<T2>(),
            "Alignments must match"
        );

        let src = ManuallyDrop::new(src);

        List {
            len: src.len,
            items: unsafe {
                std::ptr::read(
                    (&src.items as *const [MaybeUninit<T2>; N]).cast::<[MaybeUninit<T>; N]>(),
                )
            },
        }
    }
}

impl<const N: usize, T: Clone> List<N, T> {
    pub fn repeat(item: T, len: usize) -> List<N, T> {
        let mut list = List::new();
        // todo: use memcopy or something?
        // should use the extend logic below...
        iter::repeat_n(item, len).collect_into(&mut list);
        list
    }
}

impl<const N: usize, T: fmt::Debug> fmt::Debug for List<N, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<const N: usize, T> Default for List<N, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, T> Extend<T> for List<N, T> {
    // todo: implement other functions (the ones that support capacity)
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<const N: usize, T> FromIterator<T> for List<N, T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut list = Self::new();
        list.extend(iter);
        list
    }
}
