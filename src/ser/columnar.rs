//! `ColumnarValueFormat` — a column-major (Arrow-style) [`GroupFormat`] over a
//! fixed schema of fixed-width integral columns (Java `ColumnarValueFormat`,
//! spec-missing #10 / roadmap R7).
//!
//! A group of `n` fixed-arity rows is stored COLUMN-BY-COLUMN so a scan over one
//! column reads only that column's contiguous byte run, never the whole group.
//! The schema is fixed at construction and is NOT written on the wire.
//!
//! ### Wire layout (column-major)
//! For `n` rows (supplied externally by the node header) and column widths
//! `w0,w1,..`:
//! ```text
//!   [ col0 : n*w0 ][ col1 : n*w1 ] ... [ col(C-1) : n*w(C-1) ]
//! ```
//! Each cell is big-endian. With `cum_width[c] = Σ_{j<c} w_j` and
//! `row_width = Σ w_j`: cell `(row i, col c)` lives at
//! `start + n*cum_width[c] + i*w_c`, and the group ends at `start + n*row_width`.
//!
//! Byte-side operations DECODE each probed cell and compare with the same SIGNED
//! per-column order as [`compare`](ColumnarValueFormat::compare) — a raw unsigned
//! byte memcmp would misorder negative integral values.

use super::value::Value;
use super::{GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use std::cmp::Ordering;

/// Fixed-width integral column type: big-endian on the wire, signed order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Long,
    Int,
    Short,
    Byte,
}

impl ColumnType {
    /// Byte width of one cell of this column.
    pub fn width(self) -> usize {
        match self {
            ColumnType::Long => 8,
            ColumnType::Int => 4,
            ColumnType::Short => 2,
            ColumnType::Byte => 1,
        }
    }
}

// ---- cell codec (fixed-width big-endian, signed order) ----

/// Numeric widening of a cell to `i64` (signed order preserving).
#[inline]
fn cell_i64(v: &Value) -> i64 {
    match v {
        Value::Long(x) => *x,
        Value::Int(x) => *x as i64,
        Value::Short(x) => *x as i64,
        Value::Byte(x) => *x as i64,
        // A non-numeric cell is a caller schema error: columns are always one of
        // the integral `ColumnType`s. Java casts every cell to `Number` and fails
        // fast with `ClassCastException`; we panic rather than silently coerce to
        // 0, which would corrupt ordering and hide the bug.
        Value::Str(_) | Value::Bytes(_) => {
            panic!("non-numeric cell in columnar value group: {v:?}")
        }
    }
}

/// Truncate a cell to the column width (mirrors Java `intValue()/shortValue()/
/// byteValue()` before compare), preserving signed order within the width.
#[inline]
fn cell_as(t: ColumnType, v: &Value) -> i64 {
    let n = cell_i64(v);
    match t {
        ColumnType::Long => n,
        ColumnType::Int => n as i32 as i64,
        ColumnType::Short => n as i16 as i64,
        ColumnType::Byte => n as i8 as i64,
    }
}

/// Write one cell (big-endian, low bits per column width — Java `writeCell`).
#[inline]
fn write_cell(out: &mut DataOutput2, t: ColumnType, v: &Value) {
    let n = cell_i64(v);
    match t {
        ColumnType::Long => out.write_i64(n),
        ColumnType::Int => out.write_i32(n as i32),
        ColumnType::Short => out.write_i16(n as i16),
        ColumnType::Byte => out.write_u8(n as u8),
    }
}

/// Read one cell into its typed [`Value`] (Java `readCell`).
#[inline]
fn read_cell(input: &mut dyn DataInput2, t: ColumnType) -> Result<Value> {
    Ok(match t {
        ColumnType::Long => Value::Long(input.read_i64()?),
        ColumnType::Int => Value::Int(input.read_i32()?),
        ColumnType::Short => Value::Short(input.read_i16()?),
        ColumnType::Byte => Value::Byte(input.read_i8()?),
    })
}

/// Signed per-column compare (Java `compareCell`).
#[inline]
fn compare_cell(t: ColumnType, a: &Value, b: &Value) -> Ordering {
    cell_as(t, a).cmp(&cell_as(t, b))
}

/// Columnar value format over a fixed schema of fixed-width integral columns.
///
/// Element = a full-arity row (`Vec<Value>`), group = `Vec<Vec<Value>>`.
#[derive(Clone)]
pub struct ColumnarValueFormat {
    schema: Vec<ColumnType>,
    /// Length `C+1`; `cum_width[c]` = bytes of columns before `c`,
    /// `cum_width[C]` = `row_width`.
    cum_width: Vec<usize>,
    row_width: usize,
    row_ser: RowSerializer,
}

#[inline]
fn seek_overflow() -> DbError {
    DbError::corrupt("columnar seek overflow")
}

impl ColumnarValueFormat {
    /// Build a columnar value format over the given fixed-width columns
    /// (arity = length, `>= 1`).
    pub fn of(columns: &[ColumnType]) -> ColumnarValueFormat {
        assert!(!columns.is_empty(), "columnar schema needs >= 1 column");
        let schema = columns.to_vec();
        let mut cum_width = Vec::with_capacity(schema.len() + 1);
        cum_width.push(0usize);
        for c in &schema {
            let prev = *cum_width.last().unwrap();
            cum_width.push(prev + c.width());
        }
        let row_width = *cum_width.last().unwrap();
        let row_ser = RowSerializer {
            schema: schema.clone(),
            row_width,
        };
        ColumnarValueFormat {
            schema,
            cum_width,
            row_width,
            row_ser,
        }
    }

    /// Number of columns (row arity).
    pub fn column_count(&self) -> usize {
        self.schema.len()
    }
    /// Type of column `col`.
    pub fn column_type(&self, col: usize) -> ColumnType {
        self.schema[col]
    }
    /// Row width in bytes (`Σ column widths`).
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    // ---- checked seek math (a torn/oversize node must fail fast) ----

    /// Byte offset of cell `(row, col)` relative to a group starting at `start`,
    /// for a group of `size` rows. Checked against overflow (D4).
    fn cell_offset(&self, start: usize, size: usize, col: usize, row: usize) -> Result<usize> {
        let cols_before = size
            .checked_mul(self.cum_width[col])
            .ok_or_else(seek_overflow)?;
        let within = row
            .checked_mul(self.schema[col].width())
            .ok_or_else(seek_overflow)?;
        start
            .checked_add(cols_before)
            .and_then(|p| p.checked_add(within))
            .ok_or_else(seek_overflow)
    }

    /// Byte offset one past the group (`start + size*row_width`), checked.
    fn group_end(&self, start: usize, size: usize) -> Result<usize> {
        let bytes = size.checked_mul(self.row_width).ok_or_else(seek_overflow)?;
        start.checked_add(bytes).ok_or_else(seek_overflow)
    }

    #[inline]
    fn check_arity(&self, row: &[Value]) {
        assert_eq!(
            row.len(),
            self.schema.len(),
            "row/probe arity != schema arity"
        );
    }

    /// Lexicographic compare of two full-arity rows (column 0 most significant).
    fn compare_rows(&self, a: &[Value], b: &[Value]) -> Ordering {
        self.check_arity(a);
        self.check_arity(b);
        for c in 0..self.schema.len() {
            let cmp = compare_cell(self.schema[c], &a[c], &b[c]);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    /// Compare stored row `row` against the probe, column-by-column, early exit
    /// (returns `stored.cmp(key)`).
    fn compare_row_at(
        &self,
        input: &mut dyn DataInput2,
        start: usize,
        size: usize,
        row: usize,
        key: &[Value],
    ) -> Result<Ordering> {
        for c in 0..self.schema.len() {
            input.seek(self.cell_offset(start, size, c, row)?)?;
            let cell = read_cell(input, self.schema[c])?;
            let cmp = compare_cell(self.schema[c], &cell, &key[c]);
            if cmp != Ordering::Equal {
                return Ok(cmp);
            }
        }
        Ok(Ordering::Equal)
    }

    /// Cursor over ONE column's values for rows `[from, to)`: reads only that
    /// column's contiguous byte run (`n*w_col` bytes), never the whole
    /// `n*row_width` group — the columnar scan win. On exhaustion the input is
    /// left at group end (so it composes with byte-side parsing of following
    /// fields). `input` must be positioned at group start.
    pub fn column_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        size: usize,
        col: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Value> + 'a>> {
        if col >= self.schema.len() {
            return Err(DbError::corrupt("column_cursor col out of range"));
        }
        if from > to || to > size {
            return Err(DbError::corrupt("column_cursor bounds"));
        }
        let start = input.pos();
        let t = self.schema[col];
        Ok(Box::new(ColumnCursor {
            fmt: self,
            input,
            start,
            count: size,
            col,
            t,
            to,
            idx: from,
            started: false,
            cur: None,
            exhausted: false,
        }))
    }
}

impl GroupFormat for ColumnarValueFormat {
    type Elem = Vec<Value>;
    type Group = Vec<Vec<Value>>;

    fn element(&self) -> &dyn Serializer<Vec<Value>> {
        &self.row_ser
    }

    // ---- object side ----

    fn empty(&self) -> Vec<Vec<Value>> {
        Vec::new()
    }
    fn size(&self, g: &Vec<Vec<Value>>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<Vec<Value>>, pos: usize) -> Vec<Value> {
        g[pos].clone()
    }

    fn search(&self, g: &Vec<Vec<Value>>, key: &Vec<Value>) -> SearchResult {
        self.check_arity(key); // fail fast even when the group is empty
        let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            match self.compare_rows(&g[mid], key) {
                Ordering::Equal => return Ok(mid),
                Ordering::Less => lo = mid as isize + 1,
                Ordering::Greater => hi = mid as isize - 1,
            }
        }
        Err(lo as usize)
    }

    fn insert(&self, g: &Vec<Vec<Value>>, pos: usize, value: Vec<Value>) -> Vec<Vec<Value>> {
        self.check_arity(&value);
        let mut r = Vec::with_capacity(g.len() + 1);
        r.extend_from_slice(&g[..pos]);
        r.push(value);
        r.extend_from_slice(&g[pos..]);
        r
    }
    fn set(&self, g: &Vec<Vec<Value>>, pos: usize, value: Vec<Value>) -> Vec<Vec<Value>> {
        self.check_arity(&value);
        let mut r = g.clone();
        r[pos] = value;
        r
    }
    fn delete(&self, g: &Vec<Vec<Value>>, pos: usize) -> Vec<Vec<Value>> {
        let mut r = Vec::with_capacity(g.len() - 1);
        r.extend_from_slice(&g[..pos]);
        r.extend_from_slice(&g[pos + 1..]);
        r
    }
    fn copy_range(&self, g: &Vec<Vec<Value>>, from: usize, to: usize) -> Vec<Vec<Value>> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[Vec<Value>]) -> Vec<Vec<Value>> {
        for row in values {
            self.check_arity(row);
        }
        values.to_vec()
    }

    // ---- wire (column-major) ----

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<Vec<Value>>) {
        for c in 0..self.schema.len() {
            let t = self.schema[c];
            for row in g {
                write_cell(out, t, &row[c]);
            }
        }
    }

    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<Vec<Value>>> {
        let mut rows: Vec<Vec<Value>> = Vec::new();
        rows.try_reserve(count)?;
        for _ in 0..count {
            let mut row = Vec::new();
            row.try_reserve(self.schema.len())?;
            // placeholder cells, overwritten column-major below
            for _ in 0..self.schema.len() {
                row.push(Value::Byte(0));
            }
            rows.push(row);
        }
        for c in 0..self.schema.len() {
            let t = self.schema[c];
            for row in rows.iter_mut() {
                row[c] = read_cell(input, t)?;
            }
        }
        Ok(rows)
    }

    // ---- byte side ----

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_get(
        &self,
        input: &mut dyn DataInput2,
        count: usize,
        pos: usize,
    ) -> Result<Vec<Value>> {
        let start = input.pos();
        let mut row = Vec::new();
        row.try_reserve(self.schema.len())?;
        for c in 0..self.schema.len() {
            input.seek(self.cell_offset(start, count, c, pos)?)?;
            row.push(read_cell(input, self.schema[c])?);
        }
        input.seek(self.group_end(start, count)?)?;
        Ok(row)
    }

    fn binary_search(
        &self,
        key: &Vec<Value>,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        if key.len() != self.schema.len() {
            return Err(DbError::corrupt("columnar probe arity != schema"));
        }
        let start = input.pos();
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            match self.compare_row_at(input, start, count, mid, key)? {
                Ordering::Equal => {
                    found = Some(mid);
                    break;
                }
                Ordering::Less => lo = mid as isize + 1,
                Ordering::Greater => hi = mid as isize - 1,
            }
        }
        input.seek(self.group_end(start, count)?)?;
        Ok(found.map(Ok).unwrap_or(Err(lo as usize)))
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Vec<Value>> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        let start = input.pos();
        Ok(Box::new(RowCursor {
            fmt: self,
            input,
            start,
            count,
            to,
            idx: from,
            started: false,
            cur: None,
            exhausted: false,
        }))
    }
}

/// Whole-row cursor: one decode (seek each column cell) per row. On exhaustion
/// leaves input at group end.
struct RowCursor<'a> {
    fmt: &'a ColumnarValueFormat,
    input: &'a mut dyn DataInput2,
    start: usize,
    count: usize,
    to: usize,
    idx: usize,
    started: bool,
    cur: Option<Vec<Value>>,
    exhausted: bool,
}

impl<'a> GroupCursor for RowCursor<'a> {
    type Elem = Vec<Value>;

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
            self.input
                .seek(self.fmt.group_end(self.start, self.count)?)?;
            return Ok(false);
        }
        let mut row = Vec::new();
        row.try_reserve(self.fmt.schema.len())?;
        for c in 0..self.fmt.schema.len() {
            self.input
                .seek(self.fmt.cell_offset(self.start, self.count, c, self.idx)?)?;
            row.push(read_cell(self.input, self.fmt.schema[c])?);
        }
        self.cur = Some(row);
        Ok(true)
    }

    fn index(&self) -> usize {
        self.idx
    }

    fn value(&self) -> Vec<Value> {
        self.cur.clone().expect("value() before next()==true")
    }
}

/// Single-column cursor: reads only column `col`'s contiguous run.
struct ColumnCursor<'a> {
    fmt: &'a ColumnarValueFormat,
    input: &'a mut dyn DataInput2,
    start: usize,
    count: usize,
    col: usize,
    t: ColumnType,
    to: usize,
    idx: usize,
    started: bool,
    cur: Option<Value>,
    exhausted: bool,
}

impl<'a> GroupCursor for ColumnCursor<'a> {
    type Elem = Value;

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
            self.input
                .seek(self.fmt.group_end(self.start, self.count)?)?;
            return Ok(false);
        }
        self.input.seek(
            self.fmt
                .cell_offset(self.start, self.count, self.col, self.idx)?,
        )?;
        self.cur = Some(read_cell(self.input, self.t)?);
        Ok(true)
    }

    fn index(&self) -> usize {
        self.idx
    }

    fn value(&self) -> Value {
        self.cur.clone().expect("value() before next()==true")
    }
}

/// Standalone single-row codec (row-major fixed-width cells): the
/// [`GroupFormat::element`] serializer.
#[derive(Clone)]
pub struct RowSerializer {
    schema: Vec<ColumnType>,
    row_width: usize,
}

impl Serializer<Vec<Value>> for RowSerializer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<Value>) {
        assert_eq!(value.len(), self.schema.len(), "row arity != schema arity");
        for c in 0..self.schema.len() {
            write_cell(out, self.schema[c], &value[c]);
        }
    }

    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<Value>> {
        let mut row = Vec::new();
        row.try_reserve(self.schema.len())?;
        for c in 0..self.schema.len() {
            row.push(read_cell(input, self.schema[c])?);
        }
        Ok(row)
    }

    fn fixed_size(&self) -> Option<usize> {
        Some(self.row_width)
    }

    fn compare(&self, a: &Vec<Value>, b: &Vec<Value>) -> Ordering {
        assert_eq!(a.len(), self.schema.len(), "probe arity != schema arity");
        assert_eq!(b.len(), self.schema.len(), "probe arity != schema arity");
        for c in 0..self.schema.len() {
            let cmp = compare_cell(self.schema[c], &a[c], &b[c]);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }

    fn equals(&self, a: &Vec<Value>, b: &Vec<Value>) -> bool {
        self.compare(a, b) == Ordering::Equal
    }

    fn equals_by_serialized_bytes(&self) -> bool {
        true // fixed-width big-endian is canonical
    }

    fn natural_order(&self) -> bool {
        false // rows are not natural-Comparable
    }
}
