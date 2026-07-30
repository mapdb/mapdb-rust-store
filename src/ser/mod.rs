//! `ser` layer — element serializers and group formats (spec 01 §2–4).
//!
//! Two central traits:
//! - [`Serializer<A>`] — element codec + ordering + logical equality.
//! - [`GroupFormat`] — the packed key/value array of one node, owning both the
//!   representation (`Group`) and the access algorithm, in an **object side**
//!   (materialized, copy-on-write) and a **byte side** (search/get directly on
//!   serialized bytes). Uses an associated `Elem`/`Group` type (decision D2)
//!   for full monomorphization with no downcasts.
//!
//! ### Deviation from spec D2 (cursor typing)
//! The spec specifies a GAT `type Cursor<'a>` for [`GroupFormat::range_cursor`].
//! This port returns `Box<dyn GroupCursor + 'a>` instead: it achieves the same
//! type-erased, writable cursor at the API edge, avoids per-format `Cursor`
//! boilerplate and GAT-plus-`dyn`-lifetime friction, and costs only one box +
//! virtual `next()` per *cursor* (not per element) on scan paths. The format
//! itself is still fully monomorphized. Revisit under profiling if cursor
//! dispatch ever shows up.

use crate::error::Result;
use crate::io::{DataInput2, DataOutput2};
use std::cmp::Ordering;

pub mod serializers;
pub mod util;

pub mod bytearray;
pub mod columnar;
pub mod families;
pub mod int;
pub mod long;
pub mod object_array;
pub mod scalar;
pub mod string_group;
pub mod string_prefix;
pub mod tuple;
pub mod value;

pub use serializers::{
    ByteArraySer, ByteArrayUnsignedSer, CharSer, IntSer, LongSer, ShortSer, StringSer, Uuid,
    UuidSer,
};

pub use families::{
    ArraySerializer, BigDecimal, BigDecimalSer, BigInt, BigIntegerSer, BooleanArraySer, BooleanSer,
    ByteArrayNoSizeSer, ByteSer, CharArraySer, CompressionSerializer, Date, DateSer,
    DoubleArraySer, DoubleSer, FloatArraySer, FloatSer, IntArraySer, IntegerPackedSer,
    LongArraySer, LongPackedSer, RecidArraySer, RecidSer, ShortArraySer, StringAsciiSer,
    StringNoSizeSer,
};
pub use value::Value;

/// Result of a binary search: `Ok(index)` when found, `Err(insertion_point)`
/// when not. Replaces Java's `-(ins+1)` JDK convention (spec 01 §3).
pub type SearchResult = std::result::Result<usize, usize>;

/// Encode a [`SearchResult`] into the Java `int` convention
/// (`index` / `-(ins+1)`). Useful only where a ported algorithm compares
/// against the raw integer; the Rust surface uses `SearchResult` directly.
#[inline]
pub fn search_to_java(r: SearchResult) -> i64 {
    match r {
        Ok(i) => i as i64,
        Err(ins) => -(ins as i64) - 1,
    }
}

/// Decode a Java `int` binary-search result into a [`SearchResult`].
#[inline]
pub fn search_from_java(v: i64) -> SearchResult {
    if v >= 0 {
        Ok(v as usize)
    } else {
        Err((-v - 1) as usize)
    }
}

/// Element codec. Record content is produced/consumed through this; also the
/// source of ordering and logical equality (used by `compare_and_swap`).
pub trait Serializer<A> {
    /// Serialize `value` to `out`.
    fn serialize(&self, out: &mut DataOutput2, value: &A);

    /// Deserialize a value. `size` is the total bytes available, or `None` when
    /// the value is framed inside a larger record and must self-delimit
    /// (Java's `-1` sentinel).
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<A>;

    /// Fixed serialized size in bytes, or `None` if variable.
    fn fixed_size(&self) -> Option<usize> {
        None
    }

    /// Hint for output buffer sizing.
    fn size_hint(&self) -> usize {
        self.fixed_size().unwrap_or(128)
    }

    /// Total order over values, shared with `GroupFormat::search`.
    fn compare(&self, a: &A, b: &A) -> Ordering;

    /// Logical equality (CAS uses this — byte equality is wrong for
    /// non-canonical encodings).
    fn equals(&self, a: &A, b: &A) -> bool;

    /// True iff `compare` is exactly the key type's natural ordering (JDK
    /// null-comparator convention).
    fn natural_order(&self) -> bool {
        false
    }

    /// True iff serialization is canonical (equal values ⇔ byte-identical
    /// encodings). Enables in-place byte equality tests.
    fn equals_by_serialized_bytes(&self) -> bool {
        false
    }
}

/// Forward byte-side cursor over a serialized group, yielding elements in stored
/// (key) order without materializing the whole group (spec 01 §3, `GroupCursor`).
///
/// Positioning contract: the backing input is left at group END only after the
/// cursor is exhausted (after [`next`](GroupCursor::next) first returns `false`)
/// — including for an empty group/range. A caller that abandons the scan early
/// must not assume the input is positioned for following fields.
pub trait GroupCursor {
    type Elem;
    /// Advance to the next element in range; `Ok(false)` once exhausted.
    fn next(&mut self) -> Result<bool>;
    /// 0-based absolute index of the current element (valid after `next()==Ok(true)`).
    fn index(&self) -> usize;
    /// The current element (valid after `next()==Ok(true)`), owned clone.
    fn value(&self) -> Self::Elem;
}

/// Format for a group of values — the sorted key array of a btree node, the
/// first-key directory of a SortedTableMap, etc. The `Group` associated type is
/// opaque to callers; mutating ops are copy-on-write (return a new group).
///
/// No silent fallbacks: if [`supports_binary`](GroupFormat::supports_binary) is
/// false, the byte-side methods return an `Err` and the caller must deserialize.
pub trait GroupFormat {
    /// Owned element type; cloned at every point Java would let a reference
    /// escape the group (spec D2 ownership contract).
    type Elem: Clone + Send + Sync + 'static;
    /// Opaque materialized group representation.
    type Group: Clone + Send + Sync + 'static;

    /// The element serializer (ordering, equality, wire codec).
    fn element(&self) -> &dyn Serializer<Self::Elem>;

    // ---- object side ----

    fn empty(&self) -> Self::Group;
    fn size(&self, g: &Self::Group) -> usize;
    fn get(&self, g: &Self::Group, pos: usize) -> Self::Elem;
    /// Binary-search the group for `key`.
    fn search(&self, g: &Self::Group, key: &Self::Elem) -> SearchResult;

    /// Total order over elements, shared with `search`/`binary_search`. Default
    /// delegates to the element serializer; formats whose stored layout orders
    /// differently override this together with `search`/`binary_search`.
    fn compare(&self, a: &Self::Elem, b: &Self::Elem) -> Ordering {
        self.element().compare(a, b)
    }

    /// True iff this format orders by the elements' natural ordering.
    fn natural_order(&self) -> bool {
        self.element().natural_order()
    }

    fn insert(&self, g: &Self::Group, pos: usize, value: Self::Elem) -> Self::Group;
    fn set(&self, g: &Self::Group, pos: usize, value: Self::Elem) -> Self::Group;
    fn delete(&self, g: &Self::Group, pos: usize) -> Self::Group;
    fn copy_range(&self, g: &Self::Group, from: usize, to: usize) -> Self::Group;
    fn from_slice(&self, values: &[Self::Elem]) -> Self::Group;

    // ---- wire ----

    /// Write exactly the group elements; the count is stored externally by the
    /// caller (the node header).
    fn serialize(&self, out: &mut DataOutput2, g: &Self::Group);
    /// Read a group of `count` elements.
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Self::Group>;

    // ---- byte side ----

    fn supports_binary(&self) -> bool {
        false
    }

    /// Search directly in serialized bytes. Input is positioned at group start;
    /// on return it **must** be positioned at group end.
    fn binary_search(
        &self,
        _key: &Self::Elem,
        _input: &mut dyn DataInput2,
        _count: usize,
    ) -> Result<SearchResult> {
        Err(crate::error::DbError::corrupt(
            "format does not support binary access",
        ))
    }

    /// Extract one element directly from serialized bytes. Input positioned at
    /// group start; on return positioned at group end.
    fn binary_get(
        &self,
        _input: &mut dyn DataInput2,
        _count: usize,
        _pos: usize,
    ) -> Result<Self::Elem> {
        Err(crate::error::DbError::corrupt(
            "format does not support binary access",
        ))
    }

    fn supports_range_cursor(&self) -> bool {
        self.supports_binary()
    }

    /// Open a forward sequential cursor over positions `[from, to)` of the
    /// serialized group. `input` must be positioned at group start. On
    /// exhaustion the input is left at group end (spec 01 §3 positioning
    /// contract). `0 <= from <= to <= count`.
    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Self::Elem> + 'a>>;
}

/// Default range-cursor implementation built on `binary_get` (O(n·binary_get)
/// for a full scan). A correctness fallback for formats with random byte-side
/// access but no special sequential layout; formats with sequential wire layouts
/// (delta, columnar) provide their own single-pass cursor. Not usable for
/// non-binary formats — those return `Err` from `range_cursor`.
pub struct BinaryGetCursor<'a, F: GroupFormat + ?Sized> {
    format: &'a F,
    input: &'a mut dyn DataInput2,
    group_start: usize,
    count: usize,
    to: usize,
    idx: usize, // next index - 1 sentinel via wrapping; see new()
    started: bool,
    cur: Option<F::Elem>,
    exhausted: bool,
}

impl<'a, F: GroupFormat + ?Sized> BinaryGetCursor<'a, F> {
    /// `input` must be at group start. `from<=to<=count` (validated by caller).
    pub fn new(
        format: &'a F,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Self {
        let group_start = input.pos();
        Self {
            format,
            input,
            group_start,
            count,
            to,
            idx: from,
            started: false,
            cur: None,
            exhausted: false,
        }
    }
}

impl<'a, F: GroupFormat + ?Sized> GroupCursor for BinaryGetCursor<'a, F> {
    type Elem = F::Elem;

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }
        if self.started {
            self.idx += 1;
        } else {
            self.started = true;
        }
        if self.idx >= self.to {
            self.exhausted = true;
            self.cur = None;
            // Snap input to group end.
            self.input.set_pos(self.group_start);
            if self.count == 0 {
                self.format.deserialize(self.input, 0)?;
            } else {
                self.format
                    .binary_get(self.input, self.count, self.count - 1)?;
            }
            return Ok(false);
        }
        self.input.set_pos(self.group_start);
        self.cur = Some(self.format.binary_get(self.input, self.count, self.idx)?);
        Ok(true)
    }

    fn index(&self) -> usize {
        self.idx
    }

    fn value(&self) -> F::Elem {
        self.cur.clone().expect("value() before next()==true")
    }
}
