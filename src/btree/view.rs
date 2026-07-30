//! Shared navigable range/view layer (spec 03 §5), ported from Java's
//! `OrderedMapAdapter` / `OrderedNavigableView` / `ConcurrentOrderedNavigableView`
//! / `OrderedKeySet` stack. The JDK collection-interface hierarchy collapses in
//! Rust: one [`OrderedMapAdapter`] trait + one [`RangeView`] struct carrying a
//! `descending` flag.
//!
//! `RangeView` exposes an inherent API modeled on `std::collections::BTreeMap`
//! (`range`, `first_key_value`, `pop_first`, …) plus the navigable extras
//! (`lower`/`floor`/`ceiling`/`higher`, `sub_map`, `descending`). The atomic CAS
//! methods (`put_if_absent`/`replace`) are present because every adapter here is
//! concurrent; a non-concurrent adapter (a future SortedTableMap) simply would
//! not be used through those.
//!
//! Semantics preserved from Java: bound INTERSECTION never widens the parent;
//! JDK inclusivity-at-equality; inverted / exclusive-equal ranges are empty;
//! `descending()` flips a flag without touching the interval; navigation is
//! orientation-mapped (descending ceiling = backing floor, etc.). Every method
//! enforces the bounds, not just iteration.

use crate::error::Result;
use std::cmp::Ordering;

/// Narrow bridge between an ordered map and the range/navigation layer. Bounds
/// are always in the map's NATURAL (ascending) key order regardless of the
/// calling view's orientation; a `None` bound means unbounded on that side.
pub trait OrderedMapAdapter {
    type Key: Clone;
    type Val: Clone;

    fn compare(&self, a: &Self::Key, b: &Self::Key) -> Ordering;
    /// True iff keys use natural order (JDK `comparator() == null`).
    fn natural_order(&self) -> bool;
    /// Logical value equality (format equals, not `Object.equals`).
    fn value_equals(&self, a: &Self::Val, b: &Self::Val) -> bool;

    fn get(&self, k: &Self::Key) -> Result<Option<Self::Val>>;
    fn contains_key(&self, k: &Self::Key) -> Result<bool>;
    fn put(&self, k: Self::Key, v: Self::Val) -> Result<Option<Self::Val>>;
    fn remove(&self, k: &Self::Key) -> Result<Option<Self::Val>>;
    /// Atomic conditional remove (never deletes a concurrently-updated value).
    fn remove_if(&self, k: &Self::Key, v: &Self::Val) -> Result<bool>;
    fn put_if_absent(&self, k: Self::Key, v: Self::Val) -> Result<Option<Self::Val>>;
    fn replace(&self, k: &Self::Key, v: Self::Val) -> Result<Option<Self::Val>>;
    fn replace_if(&self, k: &Self::Key, ov: &Self::Val, nv: Self::Val) -> Result<bool>;

    /// Ascending entries within `[lo,hi]` honoring inclusivity; `None` = open.
    fn entry_iter_range<'a>(
        &'a self,
        lo: Option<Self::Key>,
        lo_inc: bool,
        hi: Option<Self::Key>,
        hi_inc: bool,
    ) -> Result<Box<dyn Iterator<Item = Result<(Self::Key, Self::Val)>> + 'a>>;

    /// Descending entries within `[lo,hi]`; weakly consistent, same as ascending.
    fn descending_entry_iter_range<'a>(
        &'a self,
        lo: Option<Self::Key>,
        lo_inc: bool,
        hi: Option<Self::Key>,
        hi_inc: bool,
    ) -> Result<Box<dyn Iterator<Item = Result<(Self::Key, Self::Val)>> + 'a>>;

    /// Atomically remove and return the LEAST in-range entry, or `None`.
    fn poll_first_range(
        &self,
        lo: Option<Self::Key>,
        lo_inc: bool,
        hi: Option<Self::Key>,
        hi_inc: bool,
    ) -> Result<Option<(Self::Key, Self::Val)>>;

    /// Atomically remove and return the GREATEST in-range entry, or `None`.
    fn poll_last_range(
        &self,
        lo: Option<Self::Key>,
        lo_inc: bool,
        hi: Option<Self::Key>,
        hi_inc: bool,
    ) -> Result<Option<(Self::Key, Self::Val)>>;

    fn size_long_range(
        &self,
        lo: Option<Self::Key>,
        lo_inc: bool,
        hi: Option<Self::Key>,
        hi_inc: bool,
    ) -> Result<u64>;
}

type Ent<A> = (<A as OrderedMapAdapter>::Key, <A as OrderedMapAdapter>::Val);

/// Fully-bounded, live navigable view over an [`OrderedMapAdapter`]. Cheap to
/// clone (clones the adapter, which is itself an `Arc` handle). A `descending`
/// flag reverses orientation without touching the backing interval.
pub struct RangeView<A: OrderedMapAdapter> {
    a: A,
    lo: Option<A::Key>,
    lo_inc: bool,
    hi: Option<A::Key>,
    hi_inc: bool,
    descending: bool,
}

impl<A: OrderedMapAdapter + Clone> Clone for RangeView<A> {
    fn clone(&self) -> Self {
        RangeView {
            a: self.a.clone(),
            lo: self.lo.clone(),
            lo_inc: self.lo_inc,
            hi: self.hi.clone(),
            hi_inc: self.hi_inc,
            descending: self.descending,
        }
    }
}

impl<A: OrderedMapAdapter + Clone> RangeView<A> {
    /// Full open-bounds ascending view (backs a map's whole navigable surface).
    pub fn full(a: A) -> Self {
        RangeView {
            a,
            lo: None,
            lo_inc: true,
            hi: None,
            hi_inc: true,
            descending: false,
        }
    }

    pub fn new(
        a: A,
        lo: Option<A::Key>,
        lo_inc: bool,
        hi: Option<A::Key>,
        hi_inc: bool,
        descending: bool,
    ) -> Self {
        RangeView {
            a,
            lo,
            lo_inc,
            hi,
            hi_inc,
            descending,
        }
    }

    pub fn is_descending(&self) -> bool {
        self.descending
    }

    // ---- bound predicates (backing/value order, orientation-independent) ----

    fn too_low(&self, k: &A::Key) -> bool {
        match &self.lo {
            None => false,
            Some(lo) => {
                let c = self.a.compare(k, lo);
                c == Ordering::Less || (c == Ordering::Equal && !self.lo_inc)
            }
        }
    }

    fn too_high(&self, k: &A::Key) -> bool {
        match &self.hi {
            None => false,
            Some(hi) => {
                let c = self.a.compare(k, hi);
                c == Ordering::Greater || (c == Ordering::Equal && !self.hi_inc)
            }
        }
    }

    pub fn in_range(&self, k: &A::Key) -> bool {
        !self.too_low(k) && !self.too_high(k)
    }

    /// True when `[lo2,hi2]` covers NO key: inverted (`lo2 > hi2`) or equal
    /// endpoints with either side exclusive.
    fn range_empty(
        lo2: Option<&A::Key>,
        lo_inc2: bool,
        hi2: Option<&A::Key>,
        hi_inc2: bool,
        a: &A,
    ) -> bool {
        if let (Some(l), Some(h)) = (lo2, hi2) {
            let c = a.compare(l, h);
            if c == Ordering::Greater {
                return true;
            }
            if c == Ordering::Equal {
                return !(lo_inc2 && hi_inc2);
            }
        }
        false
    }

    /// JDK-conform check for a new sub-view bound: an INCLUSIVE new bound must
    /// respect the parent's exclusivity at equality; an exclusive one only needs
    /// closed-range containment. `Err(())` = out of range.
    fn check_bound_key(&self, k: &A::Key, k_inc: bool) -> std::result::Result<(), ()> {
        if let Some(lo) = &self.lo {
            let c = self.a.compare(k, lo);
            if c == Ordering::Less || (c == Ordering::Equal && k_inc && !self.lo_inc) {
                return Err(());
            }
        }
        if let Some(hi) = &self.hi {
            let c = self.a.compare(k, hi);
            if c == Ordering::Greater || (c == Ordering::Equal && k_inc && !self.hi_inc) {
                return Err(());
            }
        }
        Ok(())
    }

    // ---- effective-bound intersection (never widen the parent) ----

    /// Effective LOWER bound = MAX of parent `(lo,lo_inc)` and probe `(k,k_inc)`.
    fn eff_lower(&self, k: &A::Key, k_inc: bool) -> (A::Key, bool) {
        match &self.lo {
            None => (k.clone(), k_inc),
            Some(lo) => match self.a.compare(k, lo) {
                Ordering::Greater => (k.clone(), k_inc),
                Ordering::Less => (lo.clone(), self.lo_inc),
                Ordering::Equal => (lo.clone(), self.lo_inc && k_inc),
            },
        }
    }

    /// Effective UPPER bound = MIN of parent `(hi,hi_inc)` and probe `(k,k_inc)`.
    fn eff_upper(&self, k: &A::Key, k_inc: bool) -> (A::Key, bool) {
        match &self.hi {
            None => (k.clone(), k_inc),
            Some(hi) => match self.a.compare(k, hi) {
                Ordering::Less => (k.clone(), k_inc),
                Ordering::Greater => (hi.clone(), self.hi_inc),
                Ordering::Equal => (hi.clone(), self.hi_inc && k_inc),
            },
        }
    }

    // ---- backing (ascending-order) navigation primitives over [lo,hi] ----

    fn first_of(
        &self,
        lo2: Option<A::Key>,
        lo_inc2: bool,
        hi2: Option<A::Key>,
        hi_inc2: bool,
    ) -> Result<Option<Ent<A>>> {
        if Self::range_empty(lo2.as_ref(), lo_inc2, hi2.as_ref(), hi_inc2, &self.a) {
            return Ok(None);
        }
        let mut it = self.a.entry_iter_range(lo2, lo_inc2, hi2, hi_inc2)?;
        match it.next() {
            None => Ok(None),
            Some(r) => Ok(Some(r?)),
        }
    }

    /// Greatest in-range entry: first element of the streaming descending
    /// iterator — one bounded descent (O(log n + leaf)), replacing the old
    /// O(range) ascending scan-keep-last.
    fn last_of(
        &self,
        lo2: Option<A::Key>,
        lo_inc2: bool,
        hi2: Option<A::Key>,
        hi_inc2: bool,
    ) -> Result<Option<Ent<A>>> {
        if Self::range_empty(lo2.as_ref(), lo_inc2, hi2.as_ref(), hi_inc2, &self.a) {
            return Ok(None);
        }
        let mut it = self
            .a
            .descending_entry_iter_range(lo2, lo_inc2, hi2, hi_inc2)?;
        match it.next() {
            None => Ok(None),
            Some(r) => Ok(Some(r?)),
        }
    }

    fn backing_ceiling(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        let (lb, li) = self.eff_lower(k, true);
        self.first_of(Some(lb), li, self.hi.clone(), self.hi_inc)
    }
    fn backing_higher(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        let (lb, li) = self.eff_lower(k, false);
        self.first_of(Some(lb), li, self.hi.clone(), self.hi_inc)
    }
    fn backing_floor(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        let (ub, ui) = self.eff_upper(k, true);
        self.last_of(self.lo.clone(), self.lo_inc, Some(ub), ui)
    }
    fn backing_lower(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        let (ub, ui) = self.eff_upper(k, false);
        self.last_of(self.lo.clone(), self.lo_inc, Some(ub), ui)
    }
    fn backing_first(&self) -> Result<Option<Ent<A>>> {
        self.first_of(self.lo.clone(), self.lo_inc, self.hi.clone(), self.hi_inc)
    }
    fn backing_last(&self) -> Result<Option<Ent<A>>> {
        self.last_of(self.lo.clone(), self.lo_inc, self.hi.clone(), self.hi_inc)
    }
    fn backing_poll_first(&self) -> Result<Option<Ent<A>>> {
        if Self::range_empty(
            self.lo.as_ref(),
            self.lo_inc,
            self.hi.as_ref(),
            self.hi_inc,
            &self.a,
        ) {
            return Ok(None);
        }
        self.a
            .poll_first_range(self.lo.clone(), self.lo_inc, self.hi.clone(), self.hi_inc)
    }
    fn backing_poll_last(&self) -> Result<Option<Ent<A>>> {
        if Self::range_empty(
            self.lo.as_ref(),
            self.lo_inc,
            self.hi.as_ref(),
            self.hi_inc,
            &self.a,
        ) {
            return Ok(None);
        }
        self.a
            .poll_last_range(self.lo.clone(), self.lo_inc, self.hi.clone(), self.hi_inc)
    }

    // ---- entry navigation (orientation-mapped, spec §D) ----

    pub fn first_entry(&self) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_last()
        } else {
            self.backing_first()
        }
    }
    pub fn last_entry(&self) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_first()
        } else {
            self.backing_last()
        }
    }
    pub fn lower_entry(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_higher(k)
        } else {
            self.backing_lower(k)
        }
    }
    pub fn floor_entry(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_ceiling(k)
        } else {
            self.backing_floor(k)
        }
    }
    pub fn ceiling_entry(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_floor(k)
        } else {
            self.backing_ceiling(k)
        }
    }
    pub fn higher_entry(&self, k: &A::Key) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_lower(k)
        } else {
            self.backing_higher(k)
        }
    }

    pub fn poll_first_entry(&self) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_poll_last()
        } else {
            self.backing_poll_first()
        }
    }
    pub fn poll_last_entry(&self) -> Result<Option<Ent<A>>> {
        if self.descending {
            self.backing_poll_first()
        } else {
            self.backing_poll_last()
        }
    }

    pub fn lower_key(&self, k: &A::Key) -> Result<Option<A::Key>> {
        Ok(self.lower_entry(k)?.map(|e| e.0))
    }
    pub fn floor_key(&self, k: &A::Key) -> Result<Option<A::Key>> {
        Ok(self.floor_entry(k)?.map(|e| e.0))
    }
    pub fn ceiling_key(&self, k: &A::Key) -> Result<Option<A::Key>> {
        Ok(self.ceiling_entry(k)?.map(|e| e.0))
    }
    pub fn higher_key(&self, k: &A::Key) -> Result<Option<A::Key>> {
        Ok(self.higher_entry(k)?.map(|e| e.0))
    }
    pub fn first_key(&self) -> Result<Option<A::Key>> {
        Ok(self.first_entry()?.map(|e| e.0))
    }
    pub fn last_key(&self) -> Result<Option<A::Key>> {
        Ok(self.last_entry()?.map(|e| e.0))
    }

    // ---- point ops (bounded, orientation-independent) ----

    pub fn get(&self, key: &A::Key) -> Result<Option<A::Val>> {
        if self.in_range(key) {
            self.a.get(key)
        } else {
            Ok(None)
        }
    }
    pub fn contains_key(&self, key: &A::Key) -> Result<bool> {
        if self.in_range(key) {
            self.a.contains_key(key)
        } else {
            Ok(false)
        }
    }
    /// Out-of-range `put` is a programming error (`Err(())` mapped to a caller
    /// panic in the map wrapper); mirrors Java's `IllegalArgumentException`.
    pub fn put(&self, key: A::Key, value: A::Val) -> Result<Option<A::Val>> {
        assert!(self.in_range(&key), "key out of submap range");
        self.a.put(key, value)
    }
    pub fn remove(&self, key: &A::Key) -> Result<Option<A::Val>> {
        if self.in_range(key) {
            self.a.remove(key)
        } else {
            Ok(None)
        }
    }
    pub fn remove_if(&self, key: &A::Key, value: &A::Val) -> Result<bool> {
        if self.in_range(key) {
            self.a.remove_if(key, value)
        } else {
            Ok(false)
        }
    }
    pub fn put_if_absent(&self, key: A::Key, value: A::Val) -> Result<Option<A::Val>> {
        assert!(self.in_range(&key), "key out of submap range");
        self.a.put_if_absent(key, value)
    }
    pub fn replace(&self, key: &A::Key, value: A::Val) -> Result<Option<A::Val>> {
        if self.in_range(key) {
            self.a.replace(key, value)
        } else {
            Ok(None)
        }
    }
    pub fn replace_if(&self, key: &A::Key, old: &A::Val, new: A::Val) -> Result<bool> {
        if self.in_range(key) {
            self.a.replace_if(key, old, new)
        } else {
            Ok(false)
        }
    }

    // ---- bulk / size ----

    pub fn size_long(&self) -> Result<u64> {
        if Self::range_empty(
            self.lo.as_ref(),
            self.lo_inc,
            self.hi.as_ref(),
            self.hi_inc,
            &self.a,
        ) {
            return Ok(0);
        }
        self.a
            .size_long_range(self.lo.clone(), self.lo_inc, self.hi.clone(), self.hi_inc)
    }

    pub fn is_empty(&self) -> Result<bool> {
        // Must surface a first-item load error, not swallow it as "non-empty".
        match self.iter()?.next() {
            None => Ok(true),
            Some(Ok(_)) => Ok(false),
            Some(Err(e)) => Err(e),
        }
    }

    /// Bounded clear: removes ONLY in-range entries (snapshots keys first).
    pub fn clear(&self) -> Result<()> {
        let mut keys = Vec::new();
        for r in self.ascending_range()? {
            keys.push(r?.0);
        }
        for k in &keys {
            self.a.remove(k)?;
        }
        Ok(())
    }

    fn ascending_range(&self) -> Result<Box<dyn Iterator<Item = Result<Ent<A>>> + '_>> {
        if Self::range_empty(
            self.lo.as_ref(),
            self.lo_inc,
            self.hi.as_ref(),
            self.hi_inc,
            &self.a,
        ) {
            return Ok(Box::new(std::iter::empty()));
        }
        self.a
            .entry_iter_range(self.lo.clone(), self.lo_inc, self.hi.clone(), self.hi_inc)
    }

    fn descending_range(&self) -> Result<Box<dyn Iterator<Item = Result<Ent<A>>> + '_>> {
        if Self::range_empty(
            self.lo.as_ref(),
            self.lo_inc,
            self.hi.as_ref(),
            self.hi_inc,
            &self.a,
        ) {
            return Ok(Box::new(std::iter::empty()));
        }
        self.a.descending_entry_iter_range(
            self.lo.clone(),
            self.lo_inc,
            self.hi.clone(),
            self.hi_inc,
        )
    }

    /// Entry iterator in THIS view's orientation.
    pub fn iter(&self) -> Result<Box<dyn Iterator<Item = Result<Ent<A>>> + '_>> {
        if self.descending {
            self.descending_range()
        } else {
            self.ascending_range()
        }
    }

    /// Key iterator in this view's orientation.
    pub fn keys(&self) -> Result<impl Iterator<Item = Result<A::Key>> + '_> {
        Ok(self.iter()?.map(|r| r.map(|e| e.0)))
    }

    /// Collect all in-range entries in orientation order (test/utility helper).
    pub fn entries(&self) -> Result<Vec<Ent<A>>> {
        let mut v = Vec::new();
        for r in self.iter()? {
            v.push(r?);
        }
        Ok(v)
    }

    // ---- descending / sub-map views ----

    /// Flip orientation without touching the interval.
    pub fn descending(&self) -> Self {
        RangeView {
            a: self.a.clone(),
            lo: self.lo.clone(),
            lo_inc: self.lo_inc,
            hi: self.hi.clone(),
            hi_inc: self.hi_inc,
            descending: !self.descending,
        }
    }

    /// Build a sub-view: each side either inherits the parent bound or checks +
    /// intersects an argument bound. `Err(())` = argument out of range.
    fn make_sub(
        &self,
        lo_arg: Option<(A::Key, bool)>,
        hi_arg: Option<(A::Key, bool)>,
    ) -> std::result::Result<Self, ()> {
        let (n_lo, n_lo_inc) = match lo_arg {
            None => (self.lo.clone(), self.lo_inc),
            Some((k, ki)) => {
                self.check_bound_key(&k, ki)?;
                let (b, bi) = self.eff_lower(&k, ki);
                (Some(b), bi)
            }
        };
        let (n_hi, n_hi_inc) = match hi_arg {
            None => (self.hi.clone(), self.hi_inc),
            Some((k, ki)) => {
                self.check_bound_key(&k, ki)?;
                let (b, bi) = self.eff_upper(&k, ki);
                (Some(b), bi)
            }
        };
        Ok(RangeView {
            a: self.a.clone(),
            lo: n_lo,
            lo_inc: n_lo_inc,
            hi: n_hi,
            hi_inc: n_hi_inc,
            descending: self.descending,
        })
    }

    /// `subMap(from, fromInc, to, toInc)` — args in THIS view's orientation.
    /// Panics on `from > to` (mirrors Java `IllegalArgumentException`).
    pub fn sub_map(&self, from: A::Key, from_inc: bool, to: A::Key, to_inc: bool) -> Self {
        let r = if !self.descending {
            assert!(
                self.a.compare(&from, &to) != Ordering::Greater,
                "fromKey > toKey"
            );
            self.make_sub(Some((from, from_inc)), Some((to, to_inc)))
        } else {
            // descending: args are in descending order (backing from >= to)
            assert!(
                self.a.compare(&to, &from) != Ordering::Greater,
                "fromKey > toKey"
            );
            self.make_sub(Some((to, to_inc)), Some((from, from_inc)))
        };
        r.expect("sub_map bound out of range")
    }

    pub fn head_map(&self, to: A::Key, inc: bool) -> Self {
        // ascending headMap = backing keys < to; descending = backing keys > to
        let r = if self.descending {
            self.make_sub(Some((to, inc)), None)
        } else {
            self.make_sub(None, Some((to, inc)))
        };
        r.expect("head_map bound out of range")
    }

    pub fn tail_map(&self, from: A::Key, inc: bool) -> Self {
        // ascending tailMap = backing keys >= from; descending = backing keys < from
        let r = if self.descending {
            self.make_sub(None, Some((from, inc)))
        } else {
            self.make_sub(Some((from, inc)), None)
        };
        r.expect("tail_map bound out of range")
    }
}
