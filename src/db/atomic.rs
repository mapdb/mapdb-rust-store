//! Atomic scalar cells over a Store4 record (Java `org.mapdb.db.Atomic`).
//!
//! Each atomic is a single store record read/written through the record's logical
//! CAS. All are cheap to clone (they share `Arc<S>` and a recid), so the DB
//! instance cache can hand back a shared-state clone (Java same-instance parity).
//!
//! `AtomicLong`/`AtomicInteger`/`AtomicBoolean` hold a non-null primitive record;
//! `AtomicString`/`AtomicVar` are nullable (a null record decodes to `None`).
//! Catalog rows: `type=AtomicLong|AtomicInteger|AtomicBoolean|AtomicString|AtomicVar`,
//! `recid=<decimal>`, and for `AtomicVar` additionally `serializer=<descriptor>`.

use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use crate::ser::families::BOOLEAN;
use crate::ser::serializers::{INT, LONG, STRING};
use crate::ser::Serializer;
use crate::store::{Recid, Store};
use std::cmp::Ordering;
use std::sync::Arc;

#[inline]
fn missing() -> DbError {
    DbError::corrupt("atomic record missing")
}

/// Nullable-string codec (Java `Serializers.STRING_NULLABLE`): a presence byte
/// (`0x00` = null, `0x01` = present) then the ordinary `STRING` encoding when
/// present. `AtomicString` always writes a (non-null) record whose *content*
/// encodes null-ness, so `get()` on a fresh `atomicString` returns `None`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullableStringSer;

/// The shared instance.
pub static STRING_NULLABLE: NullableStringSer = NullableStringSer;

impl Serializer<Option<String>> for NullableStringSer {
    fn serialize(&self, out: &mut DataOutput2, value: &Option<String>) {
        match value {
            None => out.write_u8(0),
            Some(s) => {
                out.write_u8(1);
                STRING.serialize(out, s);
            }
        }
    }
    fn deserialize(
        &self,
        input: &mut dyn DataInput2,
        _size: Option<usize>,
    ) -> Result<Option<String>> {
        let present = input.read_u8()?;
        if present == 0 {
            Ok(None)
        } else {
            Ok(Some(STRING.deserialize(input, None)?))
        }
    }
    fn compare(&self, a: &Option<String>, b: &Option<String>) -> Ordering {
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(x), Some(y)) => STRING.compare(x, y),
        }
    }
    fn equals(&self, a: &Option<String>, b: &Option<String>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

/// A 64-bit atomic long (Java `Atomic.Long`).
pub struct AtomicLong<S> {
    store: Arc<S>,
    recid: Recid,
}

impl<S> Clone for AtomicLong<S> {
    fn clone(&self) -> Self {
        AtomicLong {
            store: Arc::clone(&self.store),
            recid: self.recid,
        }
    }
}

impl<S: Store> AtomicLong<S> {
    pub fn new(store: Arc<S>, recid: Recid) -> Self {
        AtomicLong { store, recid }
    }
    pub fn recid(&self) -> Recid {
        self.recid
    }
    pub fn get(&self) -> Result<i64> {
        self.store.get(self.recid, &LONG)?.ok_or_else(missing)
    }
    pub fn set(&self, value: i64) -> Result<()> {
        self.store.update(self.recid, Some(&value), &LONG)
    }
    pub fn compare_and_set(&self, expect: i64, new: i64) -> Result<bool> {
        self.store
            .compare_and_swap(self.recid, Some(&expect), Some(&new), &LONG)
    }
    pub fn get_and_set(&self, new: i64) -> Result<i64> {
        loop {
            let cur = self.get()?;
            if self.compare_and_set(cur, new)? {
                return Ok(cur);
            }
        }
    }
    pub fn add_and_get(&self, delta: i64) -> Result<i64> {
        loop {
            let cur = self.get()?;
            let next = cur.wrapping_add(delta);
            if self.compare_and_set(cur, next)? {
                return Ok(next);
            }
        }
    }
    pub fn get_and_add(&self, delta: i64) -> Result<i64> {
        loop {
            let cur = self.get()?;
            let next = cur.wrapping_add(delta);
            if self.compare_and_set(cur, next)? {
                return Ok(cur);
            }
        }
    }
    pub fn increment_and_get(&self) -> Result<i64> {
        self.add_and_get(1)
    }
    pub fn get_and_increment(&self) -> Result<i64> {
        self.get_and_add(1)
    }
    pub fn decrement_and_get(&self) -> Result<i64> {
        self.add_and_get(-1)
    }
    pub fn get_and_decrement(&self) -> Result<i64> {
        self.get_and_add(-1)
    }
    /// Java `Number.intValue()` — narrowing cast.
    pub fn int_value(&self) -> Result<i32> {
        Ok(self.get()? as i32)
    }
    pub fn long_value(&self) -> Result<i64> {
        self.get()
    }
}

/// A 32-bit atomic integer (Java `Atomic.Integer`).
pub struct AtomicInteger<S> {
    store: Arc<S>,
    recid: Recid,
}

impl<S> Clone for AtomicInteger<S> {
    fn clone(&self) -> Self {
        AtomicInteger {
            store: Arc::clone(&self.store),
            recid: self.recid,
        }
    }
}

impl<S: Store> AtomicInteger<S> {
    pub fn new(store: Arc<S>, recid: Recid) -> Self {
        AtomicInteger { store, recid }
    }
    pub fn recid(&self) -> Recid {
        self.recid
    }
    pub fn get(&self) -> Result<i32> {
        self.store.get(self.recid, &INT)?.ok_or_else(missing)
    }
    pub fn set(&self, value: i32) -> Result<()> {
        self.store.update(self.recid, Some(&value), &INT)
    }
    pub fn compare_and_set(&self, expect: i32, new: i32) -> Result<bool> {
        self.store
            .compare_and_swap(self.recid, Some(&expect), Some(&new), &INT)
    }
    pub fn get_and_set(&self, new: i32) -> Result<i32> {
        loop {
            let cur = self.get()?;
            if self.compare_and_set(cur, new)? {
                return Ok(cur);
            }
        }
    }
    pub fn add_and_get(&self, delta: i32) -> Result<i32> {
        loop {
            let cur = self.get()?;
            let next = cur.wrapping_add(delta);
            if self.compare_and_set(cur, next)? {
                return Ok(next);
            }
        }
    }
    pub fn get_and_add(&self, delta: i32) -> Result<i32> {
        loop {
            let cur = self.get()?;
            let next = cur.wrapping_add(delta);
            if self.compare_and_set(cur, next)? {
                return Ok(cur);
            }
        }
    }
    pub fn increment_and_get(&self) -> Result<i32> {
        self.add_and_get(1)
    }
    pub fn get_and_increment(&self) -> Result<i32> {
        self.get_and_add(1)
    }
    pub fn int_value(&self) -> Result<i32> {
        self.get()
    }
    pub fn long_value(&self) -> Result<i64> {
        Ok(self.get()? as i64)
    }
}

/// An atomic boolean (Java `Atomic.Boolean`).
pub struct AtomicBoolean<S> {
    store: Arc<S>,
    recid: Recid,
}

impl<S> Clone for AtomicBoolean<S> {
    fn clone(&self) -> Self {
        AtomicBoolean {
            store: Arc::clone(&self.store),
            recid: self.recid,
        }
    }
}

impl<S: Store> AtomicBoolean<S> {
    pub fn new(store: Arc<S>, recid: Recid) -> Self {
        AtomicBoolean { store, recid }
    }
    pub fn recid(&self) -> Recid {
        self.recid
    }
    pub fn get(&self) -> Result<bool> {
        self.store.get(self.recid, &BOOLEAN)?.ok_or_else(missing)
    }
    pub fn set(&self, value: bool) -> Result<()> {
        self.store.update(self.recid, Some(&value), &BOOLEAN)
    }
    pub fn compare_and_set(&self, expect: bool, new: bool) -> Result<bool> {
        self.store
            .compare_and_swap(self.recid, Some(&expect), Some(&new), &BOOLEAN)
    }
    pub fn get_and_set(&self, new: bool) -> Result<bool> {
        loop {
            let cur = self.get()?;
            if self.compare_and_set(cur, new)? {
                return Ok(cur);
            }
        }
    }
}

/// A nullable atomic string (Java `Atomic.String`). The store record is ALWAYS
/// present; its content encodes null-ness via a leading presence byte, so
/// `create` with no initial value writes a present record whose first byte is
/// `0x00` (NOT a store-level null / deleted record), and `get()` decodes that to
/// `None`.
pub struct AtomicString<S> {
    store: Arc<S>,
    recid: Recid,
}

impl<S> Clone for AtomicString<S> {
    fn clone(&self) -> Self {
        AtomicString {
            store: Arc::clone(&self.store),
            recid: self.recid,
        }
    }
}

impl<S: Store> AtomicString<S> {
    pub fn new(store: Arc<S>, recid: Recid) -> Self {
        AtomicString { store, recid }
    }
    pub fn recid(&self) -> Recid {
        self.recid
    }
    /// `None` when the stored value is null (Java returns `null`). The record is
    /// always present; its content's presence byte encodes null-ness.
    pub fn get(&self) -> Result<Option<String>> {
        Ok(self.store.get(self.recid, &STRING_NULLABLE)?.flatten())
    }
    pub fn set(&self, value: Option<&String>) -> Result<()> {
        let owned: Option<String> = value.cloned();
        self.store
            .update(self.recid, Some(&owned), &STRING_NULLABLE)
    }
    pub fn set_str(&self, value: &str) -> Result<()> {
        let owned: Option<String> = Some(value.to_string());
        self.store
            .update(self.recid, Some(&owned), &STRING_NULLABLE)
    }
    pub fn compare_and_set(&self, expect: Option<&String>, new: Option<&String>) -> Result<bool> {
        let exp: Option<String> = expect.cloned();
        let new_owned: Option<String> = new.cloned();
        self.store
            .compare_and_swap(self.recid, Some(&exp), Some(&new_owned), &STRING_NULLABLE)
    }
}

/// A nullable atomic cell over an arbitrary element serializer (Java
/// `Atomic.Var<E>`). Cheap to clone; shares `Arc<S>`, the recid, and the
/// serializer instance.
pub struct AtomicVar<S, E, Se: Serializer<E> + Sync> {
    store: Arc<S>,
    recid: Recid,
    serializer: Arc<Se>,
    _marker: std::marker::PhantomData<fn() -> E>,
}

impl<S, E, Se: Serializer<E> + Sync> Clone for AtomicVar<S, E, Se> {
    fn clone(&self) -> Self {
        AtomicVar {
            store: Arc::clone(&self.store),
            recid: self.recid,
            serializer: Arc::clone(&self.serializer),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<S, E, Se> AtomicVar<S, E, Se>
where
    S: Store,
    E: Clone + Send + Sync + 'static,
    Se: Serializer<E> + Sync,
{
    pub fn new(store: Arc<S>, recid: Recid, serializer: Arc<Se>) -> Self {
        AtomicVar {
            store,
            recid,
            serializer,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn recid(&self) -> Recid {
        self.recid
    }
    pub fn serializer(&self) -> &Se {
        &self.serializer
    }
    /// `None` when the record is null.
    pub fn get(&self) -> Result<Option<E>> {
        self.store.get(self.recid, self.serializer.as_ref())
    }
    pub fn set(&self, value: Option<&E>) -> Result<()> {
        self.store
            .update(self.recid, value, self.serializer.as_ref())
    }
    pub fn set_value(&self, value: &E) -> Result<()> {
        self.store
            .update(self.recid, Some(value), self.serializer.as_ref())
    }
    pub fn compare_and_set(&self, expect: Option<&E>, new: Option<&E>) -> Result<bool> {
        self.store
            .compare_and_swap(self.recid, expect, new, self.serializer.as_ref())
    }
    pub fn get_and_set(&self, new: Option<&E>) -> Result<Option<E>> {
        loop {
            let cur = self.get()?;
            if self.store.compare_and_swap(
                self.recid,
                cur.as_ref(),
                new,
                self.serializer.as_ref(),
            )? {
                return Ok(cur);
            }
        }
    }
}
