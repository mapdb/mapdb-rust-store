#![allow(private_bounds)]
//! Map-backed navigable set (Java `DB.treeSet`) built on a `BTreeMap` whose value
//! format serializes nothing (`NoValueFormat`, Java `Serializers.NO_VALUE` inside
//! an `ObjectArrayFormat`). Only the `TreeSet` catalog row is written; the value
//! format is implicit and has no descriptor.

use crate::btree::{BTreeMap, RangeView};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use crate::ser::{GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::store::{Store, StoreLease};
use std::cmp::Ordering;

/// The element serializer for the absent value: reads/writes zero bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoValueSer;

impl Serializer<()> for NoValueSer {
    fn serialize(&self, _out: &mut DataOutput2, _value: &()) {}
    fn deserialize(&self, _input: &mut dyn DataInput2, _size: Option<usize>) -> Result<()> {
        Ok(())
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(0)
    }
    fn compare(&self, _a: &(), _b: &()) -> Ordering {
        Ordering::Equal
    }
    fn equals(&self, _a: &(), _b: &()) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

/// A value group format that carries only a count and serializes to zero bytes
/// (Java map-backed set's no-value format). The group is the element count; every
/// element is `()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoValueFormat;

impl GroupFormat for NoValueFormat {
    type Elem = ();
    type Group = usize;

    fn element(&self) -> &dyn Serializer<()> {
        &NoValueSer
    }
    fn empty(&self) -> usize {
        0
    }
    fn size(&self, g: &usize) -> usize {
        *g
    }
    fn get(&self, _g: &usize, _pos: usize) {}
    fn search(&self, _g: &usize, _key: &()) -> SearchResult {
        // Values are never searched.
        Err(0)
    }
    fn insert(&self, g: &usize, _pos: usize, _value: ()) -> usize {
        g + 1
    }
    fn set(&self, g: &usize, _pos: usize, _value: ()) -> usize {
        *g
    }
    fn delete(&self, g: &usize, _pos: usize) -> usize {
        g.saturating_sub(1)
    }
    fn copy_range(&self, _g: &usize, from: usize, to: usize) -> usize {
        to - from
    }
    fn from_slice(&self, values: &[()]) -> usize {
        values.len()
    }
    fn serialize(&self, _out: &mut DataOutput2, _g: &usize) {}
    fn deserialize(&self, _input: &mut dyn DataInput2, count: usize) -> Result<usize> {
        Ok(count)
    }
    fn range_cursor<'a>(
        &'a self,
        _input: &'a mut dyn DataInput2,
        _count: usize,
        _from: usize,
        _to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = ()> + 'a>> {
        // Sets iterate keys via the materialized object side, never the value
        // byte cursor. This is unreachable for the set surface.
        Err(DbError::Unsupported(
            "no-value format has no byte-side cursor",
        ))
    }
}

/// A navigable set backed by a `BTreeMap<S, KF, NoValueFormat>` (Java
/// `DB.treeSet` result). Cheap to clone; shares the backing map's state.
pub struct NavigableSet<S, KF: GroupFormat> {
    map: BTreeMap<S, KF, NoValueFormat>,
}

impl<S, KF: GroupFormat> Clone for NavigableSet<S, KF> {
    fn clone(&self) -> Self {
        NavigableSet {
            map: self.map.clone(),
        }
    }
}

impl<S, KF> NavigableSet<S, KF>
where
    S: Store + StoreLease + 'static,
    KF: GroupFormat + Send + Sync + 'static,
{
    pub(crate) fn from_map(map: BTreeMap<S, KF, NoValueFormat>) -> Self {
        NavigableSet { map }
    }

    /// The backing map (for the DB facade's accessors).
    pub fn backing_map(&self) -> &BTreeMap<S, KF, NoValueFormat> {
        &self.map
    }

    pub fn counter_recid(&self) -> u64 {
        self.map.counter_recid()
    }

    pub fn root_recid_recid(&self) -> u64 {
        self.map.root_recid_recid()
    }

    /// Add `element`; returns `true` if it was not already present (Java `add`).
    pub fn add(&self, element: KF::Elem) -> Result<bool> {
        Ok(self.map.put_if_absent(element, ())?.is_none())
    }

    pub fn contains(&self, element: &KF::Elem) -> Result<bool> {
        self.map.contains_key(element)
    }

    /// Remove `element`; returns `true` if it was present (Java `remove`).
    pub fn remove(&self, element: &KF::Elem) -> Result<bool> {
        self.map.remove_only(element)
    }

    pub fn size_long(&self) -> Result<u64> {
        self.map.size_long()
    }

    pub fn is_empty(&self) -> Result<bool> {
        self.map.is_empty()
    }

    pub fn clear(&self) -> Result<()> {
        self.map.clear()
    }

    pub fn first(&self) -> Result<Option<KF::Elem>> {
        Ok(self.map.first_entry()?.map(|(k, _)| k))
    }

    pub fn last(&self) -> Result<Option<KF::Elem>> {
        Ok(self.map.last_entry()?.map(|(k, _)| k))
    }

    /// Ascending iteration over the elements.
    pub fn to_vec(&self) -> Result<Vec<KF::Elem>> {
        Ok(self.map.entries()?.into_iter().map(|(k, _)| k).collect())
    }

    // ---- NavigableSet navigation surface (Java `MapBackedNavigableSet`) ----
    //
    // Each delegates to the corresponding backing-map navigation primitive and
    // drops the (always-`()`) value. Bounded views are materialized snapshots,
    // consistent with `to_vec`.

    /// Greatest element strictly less than `e` (Java `lower`).
    pub fn lower(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        Ok(self.map.lower_entry(e)?.map(|(k, _)| k))
    }

    /// Greatest element less than or equal to `e` (Java `floor`).
    pub fn floor(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        Ok(self.map.floor_entry(e)?.map(|(k, _)| k))
    }

    /// Least element greater than or equal to `e` (Java `ceiling`).
    pub fn ceiling(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        Ok(self.map.ceiling_entry(e)?.map(|(k, _)| k))
    }

    /// Least element strictly greater than `e` (Java `higher`).
    pub fn higher(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        Ok(self.map.higher_entry(e)?.map(|(k, _)| k))
    }

    /// Atomically remove and return the least element, or `None` when empty
    /// (Java `pollFirst`). Weakly-consistent selection, per the backing map.
    pub fn poll_first(&self) -> Result<Option<KF::Elem>> {
        Ok(self
            .map
            .poll_first_entry(None, true, None, true)?
            .map(|(k, _)| k))
    }

    /// Atomically remove and return the greatest element, or `None` when empty
    /// (Java `pollLast`).
    pub fn poll_last(&self) -> Result<Option<KF::Elem>> {
        Ok(self
            .map
            .poll_last_entry(None, true, None, true)?
            .map(|(k, _)| k))
    }

    /// Descending (greatest-first) materialized snapshot. A convenience over
    /// [`descending_set`](Self::descending_set)`().to_vec()`.
    pub fn descending_to_vec(&self) -> Result<Vec<KF::Elem>> {
        self.descending_set().to_vec()
    }

    /// Live descending set view (Java `descendingSet` / `descendingIterator`).
    pub fn descending_set(&self) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.map.descending(),
        }
    }

    /// Live `[from, to]` sub-set view honoring the inclusivity flags (Java
    /// `subSet`). Panics on `from > to`, mirroring Java `IllegalArgumentException`.
    pub fn sub_set(
        &self,
        from: KF::Elem,
        from_inc: bool,
        to: KF::Elem,
        to_inc: bool,
    ) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.map.sub_map(from, from_inc, to, to_inc),
        }
    }

    /// Live view of the elements up to `to` (Java `headSet`).
    pub fn head_set(&self, to: KF::Elem, inc: bool) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.map.head_map(to, inc),
        }
    }

    /// Live view of the elements from `from` onward (Java `tailSet`).
    pub fn tail_set(&self, from: KF::Elem, inc: bool) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.map.tail_map(from, inc),
        }
    }

    pub fn is_closed(&self) -> bool {
        self.map.is_closed()
    }
}

/// A live, bounded, orientation-aware view over a [`NavigableSet`] (Java's
/// `NavigableSet` sub/head/tail/descending views). Wraps the backing map's
/// [`RangeView`], so it reflects concurrent backing-set changes and its mutators
/// (`remove`, `clear`, `poll_*`) write through to the backing set. It is
/// read/remove-only: there is no `add`, matching Java's out-of-range-add ban on
/// range views. Cheap to clone (clones the underlying `Arc` handle).
pub struct NavigableSetView<S, KF>
where
    S: Store + StoreLease + 'static,
    KF: GroupFormat + Send + Sync + 'static,
{
    view: RangeView<BTreeMap<S, KF, NoValueFormat>>,
}

impl<S, KF> Clone for NavigableSetView<S, KF>
where
    S: Store + StoreLease + 'static,
    KF: GroupFormat + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        NavigableSetView {
            view: self.view.clone(),
        }
    }
}

impl<S, KF> NavigableSetView<S, KF>
where
    S: Store + StoreLease + 'static,
    KF: GroupFormat + Send + Sync + 'static,
{
    /// Whether `element` is present within this view's bounds.
    pub fn contains(&self, element: &KF::Elem) -> Result<bool> {
        self.view.contains_key(element)
    }

    /// Remove `element` if it is present within this view's bounds; returns
    /// `true` if it was removed. Writes through to the backing set.
    pub fn remove(&self, element: &KF::Elem) -> Result<bool> {
        Ok(self.view.remove(element)?.is_some())
    }

    /// Remove every element within this view's bounds from the backing set.
    pub fn clear(&self) -> Result<()> {
        self.view.clear()
    }

    pub fn size_long(&self) -> Result<u64> {
        self.view.size_long()
    }

    pub fn is_empty(&self) -> Result<bool> {
        self.view.is_empty()
    }

    /// First element in this view's orientation (least for an ascending view,
    /// greatest for a descending one).
    pub fn first(&self) -> Result<Option<KF::Elem>> {
        self.view.first_key()
    }

    /// Last element in this view's orientation.
    pub fn last(&self) -> Result<Option<KF::Elem>> {
        self.view.last_key()
    }

    /// Greatest element (in this view's orientation) strictly before `e`.
    pub fn lower(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        self.view.lower_key(e)
    }

    /// Greatest element (in this view's orientation) at or before `e`.
    pub fn floor(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        self.view.floor_key(e)
    }

    /// Least element (in this view's orientation) at or after `e`.
    pub fn ceiling(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        self.view.ceiling_key(e)
    }

    /// Least element (in this view's orientation) strictly after `e`.
    pub fn higher(&self, e: &KF::Elem) -> Result<Option<KF::Elem>> {
        self.view.higher_key(e)
    }

    /// Remove and return the first element in this view's orientation.
    pub fn poll_first(&self) -> Result<Option<KF::Elem>> {
        Ok(self.view.poll_first_entry()?.map(|(k, _)| k))
    }

    /// Remove and return the last element in this view's orientation.
    pub fn poll_last(&self) -> Result<Option<KF::Elem>> {
        Ok(self.view.poll_last_entry()?.map(|(k, _)| k))
    }

    /// Materialized snapshot of this view's elements in orientation order.
    pub fn to_vec(&self) -> Result<Vec<KF::Elem>> {
        let mut v = Vec::new();
        for r in self.view.keys()? {
            v.push(r?);
        }
        Ok(v)
    }

    /// Live descending view of these elements (flips orientation).
    pub fn descending_set(&self) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.view.descending(),
        }
    }

    /// Live nested `[from, to]` sub-view; args are in THIS view's orientation.
    pub fn sub_set(
        &self,
        from: KF::Elem,
        from_inc: bool,
        to: KF::Elem,
        to_inc: bool,
    ) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.view.sub_map(from, from_inc, to, to_inc),
        }
    }

    /// Live nested head view (elements up to `to` in this view's orientation).
    pub fn head_set(&self, to: KF::Elem, inc: bool) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.view.head_map(to, inc),
        }
    }

    /// Live nested tail view (elements from `from` onward in this orientation).
    pub fn tail_set(&self, from: KF::Elem, inc: bool) -> NavigableSetView<S, KF> {
        NavigableSetView {
            view: self.view.tail_map(from, inc),
        }
    }
}
