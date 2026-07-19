use core::slice;
use std::{
    any::type_name,
    cmp::max,
    fmt::{self, Debug},
    iter,
    marker::PhantomData,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Bound, Index, IndexMut, IntoBounds, RangeBounds},
    ptr,
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
    pub fn new<B: IntoBounds<T>>(value: T, range: B) -> Self { Self { value, range: Bounds::new(range) } }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{value} was not in the set of expected values {expected:?}")]
pub struct ValueOutOfSetError<T: Debug + 'static> {
    pub value: T,
    pub expected: &'static [T],
}

impl<T: Debug + 'static> ValueOutOfSetError<T> {
    pub fn new(value: T, set: &'static [T]) -> Self { Self { value, expected: set } }
}

#[derive(Debug, Error, PartialEq)]
#[error("The specified value {value} was not a valid value.")]
pub struct InvalidValueError<T: Debug> {
    pub value: T,
}

impl<T: Debug> InvalidValueError<T> {
    pub fn new(value: T) -> Self { Self { value } }
}

#[derive(Debug, Error, Default)]
#[error("Unexpected end of token stream. Expected a {} instead.", type_name::<T>())]
pub struct MissingTokenError<T: Default> {
    _t: PhantomData<T>,
}

impl<T: Default> MissingTokenError<T> {
    pub fn new() -> Self { Self::default() }
}

#[derive(Debug, Error)]
#[error("Expected a {expected_token_name}, but got <something else>.")]
pub struct UnexpectedTokenError {
    pub expected_token_name: &'static str,
}

impl UnexpectedTokenError {
    pub fn new(token_name: &'static str) -> Self { Self { expected_token_name: token_name } }
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
    pub fn off() -> Self { Self(Arc::new(AtomicBool::new(false))) }

    pub fn on() -> Self { Self(Arc::new(AtomicBool::new(true))) }

    pub fn get(&self) -> bool { self.0.load(Ordering::Relaxed) }

    pub fn set(&self, val: bool) { self.0.store(val, Ordering::Relaxed); }
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

    pub fn cancel(&self) { self.v.store(true, Ordering::Relaxed) }

    pub fn is_cancelled(&self) -> bool { self.v.load(Ordering::Relaxed) }
}

pub type CheckHealthResult<E> = Result<(), E>;

pub trait CheckHealth {
    type Error;

    #[must_use = "The result of a health check should be checked."]
    fn check_health(&self) -> CheckHealthResult<Self::Error>;
}

#[repr(C)]
pub struct List<const N: usize, T> {
    items: [MaybeUninit<T>; N],
    len: usize,
}

impl<const N: usize, T: PartialEq> PartialEq for List<N, T> {
    fn eq(&self, other: &Self) -> bool { self.as_slice() == other.as_slice() }
}

impl<const N: usize, T: Clone> Clone for List<N, T> {
    fn clone(&self) -> Self {
        let mut new_list = Self::new();
        new_list.len = self.len;
        // todo: use memcopy?
        for (i, item) in self.as_slice().iter().enumerate() {
            unsafe {
                new_list.items.get_unchecked_mut(i).write(item.clone());
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
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub fn clear(&mut self) { self.len = 0; }

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

    #[inline]
    pub fn as_mut_subslice<R: RangeBounds<usize>>(&mut self, range: R) -> &mut [T] {
        // Resolve the exact start index
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };

        // Resolve the exact end index
        let end = match range.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len,
        };

        debug_assert!(start <= end, "range start must be <= end");
        debug_assert!(end <= self.len, "range end out of bounds");

        // SAFETY: The range is within [0, self.len). The pointer arithmetic yields a
        // valid pointer to the first element, and the length is exactly the number of
        // contiguous initialized elements.
        unsafe {
            let ptr = self.items.as_mut_ptr().cast::<T>().add(start);
            slice::from_raw_parts_mut(ptr, end - start)
        }
    }

    /// Returns a slice of the initialized elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: We only cast the slice up to `self.len`, which we
        // guarantee has been initialized via the `push` method.
        unsafe { slice::from_raw_parts(self.items.as_ptr().cast(), self.len) }
    }

    #[inline]
    pub fn as_subslice(&self, range: impl RangeBounds<usize>) -> &[T] {
        // Resolve the exact start index
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };

        // Resolve the exact end index
        let end = match range.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => self.len,
        };

        debug_assert!(start <= end, "range start must be <= end");
        debug_assert!(end <= self.len, "range end out of bounds");

        // SAFETY: The range is within [0, self.len). The pointer arithmetic yields a
        // valid pointer to the first element, and the length is exactly the number of
        // contiguous initialized elements.
        unsafe {
            let ptr = self.items.as_ptr().cast::<T>().add(start);
            slice::from_raw_parts(ptr, end - start)
        }
    }

    /// Returns an iterator over the initialized elements.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, T> { self.as_slice().iter() }

    /// Returns a mutable iterator over the initialized elements.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> { self.as_mut_slice().iter_mut() }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: We only access the element if `index` is less than `self.len`, which
            // guarantees that it has been initialized via the `push` method.
            Some(unsafe { self.items.get_unchecked(index).assume_init_ref() })
        }
        else {
            None
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that `index` is less than `self.len`.
    #[inline]
    pub const unsafe fn get_unchecked(&self, index: usize) -> &T {
        // SAFETY: The caller must ensure that `index` is less than `self.len`, which
        // guarantees that the element has been initialized via the `push`
        // method.
        unsafe { self.items.get_unchecked(index).assume_init_ref() }
    }

    /// # Safety
    ///
    /// The caller must ensure that `index` is less than `self.len`.
    #[inline]
    pub const unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: The caller must ensure that `index` is less than `self.len`, which
        // guarantees that the element has been initialized via the `push`
        // method.
        unsafe { self.items.get_unchecked_mut(index).assume_init_mut() }
    }

    /// # Safety
    ///
    /// The caller must ensure that `index` is less than `self.len`.
    #[inline]
    pub unsafe fn read_unchecked(&self, index: usize) -> T {
        // SAFETY: The caller must ensure that `index` is less than `self.len`, which
        // guarantees that the element has been initialized via the `push`
        // method.
        unsafe { self.items.get_unchecked(index).assume_init_read() }
    }

    /// # Safety
    /// The caller must ensure that `T` and `T2` have the exact same size and
    /// alignment, and that the bitwise representation of `T2` is a valid
    /// representation of `T`.
    pub unsafe fn transmute<T2>(src: List<N, T2>) -> List<N, T> {
        debug_assert_eq!(mem::size_of::<T>(), mem::size_of::<T2>(), "Sizes must match");
        debug_assert_eq!(mem::align_of::<T>(), mem::align_of::<T2>(), "Alignments must match");

        let src = ManuallyDrop::new(src);

        List {
            len: src.len,
            items: unsafe { ptr::read((&src.items as *const [MaybeUninit<T2>; N]).cast::<[MaybeUninit<T>; N]>()) },
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = T> {
        struct Drain<'a, const N: usize, T> {
            len: usize,
            list: &'a mut List<N, T>,
            index: usize,
        }

        impl<'a, const N: usize, T> Iterator for Drain<'a, N, T> {
            type Item = T;
            fn next(&mut self) -> Option<Self::Item> {
                if self.index < self.len {
                    let item = unsafe { self.list.read_unchecked(self.index) };
                    self.index += 1;
                    Some(item)
                }
                else {
                    None
                }
            }
        }

        // after this function has finished, all items will be removed
        let len = self.len;
        self.len = 0;

        Drain { list: self, len, index: 0 }
    }

    /// Copies the slice src into dest range, extending the list if necessary.
    ///
    /// # Examples
    ///
    /// ### Appending to the end of a list
    /// ```
    /// use engine::misc::List;
    ///
    /// let mut list: List<10, i32> = List::new();
    /// list.push(5);
    ///
    /// // Appending multiple items starting right after index 0
    /// let extra_moves = [12, 15, 18];
    /// list.extend_from_slice(1.., &extra_moves);
    ///
    /// assert_eq!(list.as_slice(), &[5, 12, 15, 18]);
    /// assert_eq!(list.len(), 4);
    /// ```
    ///
    /// ### Overwriting an existing range
    /// ```
    /// use engine::misc::List;
    ///
    /// let mut list: List<5, &'static str> = List::new();
    /// list.push("e2e4");
    /// list.push("e7e5");
    /// list.push("g1f3");
    /// list.push("g8f6");
    ///
    /// // Replace "e7e5" and "g1f3" with new moves
    /// let corrections = ["c7c5", "b1c3"];
    /// list.extend_from_slice(1..3, &corrections);
    ///
    /// assert_eq!(list.as_slice(), &["e2e4", "c7c5", "b1c3", "g8f6"]);
    /// ```
    pub fn extend_from_slice(&mut self, dest: impl RangeBounds<usize>, src: &[T])
    where
        T: Copy,
    {
        if src.len() == 0 {
            return;
        }

        // Resolve the exact start index
        let start = match dest.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };

        // Resolve the exact end index
        let end = match dest.end_bound() {
            Bound::Included(&e) => e + 1,
            Bound::Excluded(&e) => e,
            Bound::Unbounded => start + src.len(),
        };

        debug_assert!(start <= end, "range start must be <= end");
        debug_assert_eq!(end - start, src.len(), "Source slice length must match the destination range length");

        unsafe {
            let ptr = self.items.as_mut_ptr().cast::<T>().add(start);
            ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
        }

        let new_len = max(self.len, end);
        debug_assert!(new_len <= N, "List capacity exceeded");

        // Updat length
        self.len = new_len;
    }
}

const impl<const N: usize, T> Index<usize> for List<N, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len, "Index out of bounds");

        // SAFETY: The bounds check is handled by the caller or by design constraints
        // (like the maximum 218 legal chess moves).
        unsafe { self.get_unchecked(index) }
    }
}

const impl<const N: usize, T> IndexMut<usize> for List<N, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len, "Index out of bounds");

        // SAFETY: The bounds check is handled by the caller or by design constraints
        // (like the maximum 218 legal chess moves).
        unsafe { self.get_unchecked_mut(index) }
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.debug_list().entries(self.as_slice()).finish() }
}

impl<const N: usize, T> Default for List<N, T> {
    fn default() -> Self { Self::new() }
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

impl<const N: usize, T> AsRef<[T]> for List<N, T> {
    fn as_ref(&self) -> &[T] { self.as_slice() }
}

// stdlib is not const
#[inline(always)]
pub const fn c_abs(this: i8) -> i8 { if this.is_negative() { -this } else { this } }
