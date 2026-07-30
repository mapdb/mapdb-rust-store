//! `ObjectArrayFormat<A>` — generic fallback group format over any element
//! serializer (Java `ObjectArrayFormat`). Declares `supports_binary() == false`:
//! callers must deserialize before searching (no silent fallback, spec R6).

use super::{GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use std::cmp::Ordering;

/// Group backed by `Vec<A>`, elements encoded by the element serializer.
#[derive(Clone)]
pub struct ObjectArrayFormat<A, S: Serializer<A>> {
    element: S,
    _marker: std::marker::PhantomData<fn() -> A>,
}

impl<A, S: Serializer<A>> ObjectArrayFormat<A, S> {
    pub fn new(element: S) -> Self {
        Self {
            element,
            _marker: std::marker::PhantomData,
        }
    }

    /// The concrete element serializer (for catalog-descriptor introspection).
    pub fn element_serializer(&self) -> &S {
        &self.element
    }
}

impl<A, S> GroupFormat for ObjectArrayFormat<A, S>
where
    A: Clone + Send + Sync + 'static,
    S: Serializer<A> + Send + Sync + 'static,
{
    type Elem = A;
    type Group = Vec<A>;

    fn element(&self) -> &dyn Serializer<A> {
        &self.element
    }
    fn empty(&self) -> Vec<A> {
        Vec::new()
    }
    fn size(&self, g: &Vec<A>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<A>, pos: usize) -> A {
        g[pos].clone()
    }
    fn search(&self, g: &Vec<A>, key: &A) -> SearchResult {
        let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            match self.element.compare(&g[mid], key) {
                Ordering::Equal => return Ok(mid),
                Ordering::Less => lo = mid as isize + 1,
                Ordering::Greater => hi = mid as isize - 1,
            }
        }
        Err(lo as usize)
    }
    fn insert(&self, g: &Vec<A>, pos: usize, v: A) -> Vec<A> {
        let mut r = Vec::with_capacity(g.len() + 1);
        r.extend_from_slice(&g[..pos]);
        r.push(v);
        r.extend_from_slice(&g[pos..]);
        r
    }
    fn set(&self, g: &Vec<A>, pos: usize, v: A) -> Vec<A> {
        let mut r = g.clone();
        r[pos] = v;
        r
    }
    fn delete(&self, g: &Vec<A>, pos: usize) -> Vec<A> {
        let mut r = Vec::with_capacity(g.len() - 1);
        r.extend_from_slice(&g[..pos]);
        r.extend_from_slice(&g[pos + 1..]);
        r
    }
    fn copy_range(&self, g: &Vec<A>, from: usize, to: usize) -> Vec<A> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[A]) -> Vec<A> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<A>) {
        for o in g {
            self.element.serialize(out, o);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<A>> {
        let mut r = Vec::new();
        r.try_reserve(count)?;
        for _ in 0..count {
            r.push(self.element.deserialize(input, None)?);
        }
        Ok(r)
    }

    // supports_binary == false (default); byte-side methods return Err.

    fn range_cursor<'a>(
        &'a self,
        _input: &'a mut dyn DataInput2,
        _count: usize,
        _from: usize,
        _to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = A> + 'a>> {
        Err(DbError::corrupt("format does not support range cursor"))
    }
}
