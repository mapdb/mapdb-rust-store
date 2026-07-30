//! `ColumnarValueFormat` validation: column-major wire round-trip, object-side
//! `search` vs byte-side `binary_search` coherence, `binary_get == get`,
//! positioning contract, signed order across all column widths, multi-column
//! lexicographic order, and the single-column scan (reads only that column's run).

use mapdb_rust_store::error::Result;
use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::columnar::{ColumnType, ColumnarValueFormat};
use mapdb_rust_store::ser::value::Value;
use mapdb_rust_store::ser::{GroupFormat, SearchResult};

type Row = Vec<Value>;

fn ser(f: &ColumnarValueFormat, g: &Vec<Row>) -> Vec<u8> {
    let mut out = DataOutput2::new();
    f.serialize(&mut out, g);
    out.into_vec()
}

/// A `DataInput2` that counts the bytes actually READ (via `read_u8`/`read_fully`)
/// — seeks do not count. All wider reads route through `read_u8`'s default impls,
/// so this precisely measures how many payload bytes a scan touches.
struct CountingInput<'a> {
    buf: &'a [u8],
    pos: usize,
    reads: usize,
}

impl<'a> CountingInput<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            reads: 0,
        }
    }
}

impl<'a> DataInput2 for CountingInput<'a> {
    fn len(&self) -> usize {
        self.buf.len()
    }
    fn pos(&self) -> usize {
        self.pos
    }
    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }
    fn read_u8(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| mapdb_rust_store::error::DbError::corrupt("read past end"))?;
        self.pos += 1;
        self.reads += 1;
        Ok(b)
    }
    fn read_fully(&mut self, dst: &mut [u8]) -> Result<()> {
        let end = self.pos + dst.len();
        let src = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| mapdb_rust_store::error::DbError::corrupt("read_fully past end"))?;
        dst.copy_from_slice(src);
        self.pos = end;
        self.reads += dst.len();
        Ok(())
    }
}

/// Mirror of `ser_reference::check_coherence` for the columnar format.
fn check_coherence(f: &ColumnarValueFormat, g: &Vec<Row>, probes: &[Row]) {
    let count = f.size(g);
    let bytes = ser(f, g);
    for key in probes {
        let obj: SearchResult = f.search(g, key);
        assert!(f.supports_binary());
        let mut inp = SliceInput::new(&bytes);
        let byte_res = f.binary_search(key, &mut inp, count).unwrap();
        assert_eq!(obj, byte_res, "coherence for key {key:?}");
        assert_eq!(inp.pos(), bytes.len(), "binary_search leaves at group end");
    }
    // binary_get of every position round-trips get(), and leaves input at end.
    for pos in 0..count {
        let mut ig = SliceInput::new(&bytes);
        let v = f.binary_get(&mut ig, count, pos).unwrap();
        assert_eq!(v, f.get(g, pos), "binary_get pos {pos}");
        assert_eq!(ig.pos(), bytes.len(), "binary_get leaves at group end");
    }
}

#[test]
fn column_major_wire_roundtrip() {
    // schema: LONG, SHORT  (row width 10). 3 rows.
    let f = ColumnarValueFormat::of(&[ColumnType::Long, ColumnType::Short]);
    let g: Vec<Row> = vec![
        vec![Value::Long(1), Value::Short(10)],
        vec![Value::Long(2), Value::Short(20)],
        vec![Value::Long(3), Value::Short(30)],
    ];
    let bytes = ser(&f, &g);
    // column-major: 3 longs (24 bytes) then 3 shorts (6 bytes) = 30 bytes.
    assert_eq!(bytes.len(), 30);
    assert_eq!(&bytes[0..8], &1i64.to_be_bytes());
    assert_eq!(&bytes[8..16], &2i64.to_be_bytes());
    assert_eq!(&bytes[16..24], &3i64.to_be_bytes());
    assert_eq!(&bytes[24..26], &10i16.to_be_bytes());
    assert_eq!(&bytes[26..28], &20i16.to_be_bytes());
    assert_eq!(&bytes[28..30], &30i16.to_be_bytes());

    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g);
    assert_eq!(inp.pos(), bytes.len());
}

#[test]
fn negative_cells_signed_order_all_widths() {
    // Each column type carries negative values; the group is sorted lexicographically
    // by the SIGNED per-column order. A raw unsigned memcmp would put negatives last.
    let f = ColumnarValueFormat::of(&[
        ColumnType::Long,
        ColumnType::Int,
        ColumnType::Short,
        ColumnType::Byte,
    ]);
    let mut g: Vec<Row> = vec![
        vec![
            Value::Long(i64::MIN),
            Value::Int(0),
            Value::Short(0),
            Value::Byte(0),
        ],
        vec![
            Value::Long(-1),
            Value::Int(i32::MIN),
            Value::Short(-1),
            Value::Byte(-128),
        ],
        vec![
            Value::Long(-1),
            Value::Int(i32::MIN),
            Value::Short(-1),
            Value::Byte(-1),
        ],
        vec![
            Value::Long(-1),
            Value::Int(-1),
            Value::Short(i16::MIN),
            Value::Byte(0),
        ],
        vec![
            Value::Long(0),
            Value::Int(0),
            Value::Short(0),
            Value::Byte(0),
        ],
        vec![
            Value::Long(1),
            Value::Int(-1),
            Value::Short(-1),
            Value::Byte(-1),
        ],
        vec![
            Value::Long(i64::MAX),
            Value::Int(i32::MAX),
            Value::Short(i16::MAX),
            Value::Byte(127),
        ],
    ];
    g.sort_by(|a, b| f.element().compare(a, b));
    // sanity: sorting must not have moved i64::MIN off the front nor MAX off the back.
    assert_eq!(g[0][0], Value::Long(i64::MIN));
    assert_eq!(g[g.len() - 1][0], Value::Long(i64::MAX));

    let probes: Vec<Row> = vec![
        vec![
            Value::Long(i64::MIN),
            Value::Int(0),
            Value::Short(0),
            Value::Byte(0),
        ],
        vec![
            Value::Long(-1),
            Value::Int(i32::MIN),
            Value::Short(-1),
            Value::Byte(-1),
        ],
        vec![
            Value::Long(-1),
            Value::Int(-2),
            Value::Short(0),
            Value::Byte(0),
        ],
        vec![
            Value::Long(0),
            Value::Int(0),
            Value::Short(0),
            Value::Byte(1),
        ],
        vec![
            Value::Long(i64::MAX),
            Value::Int(i32::MAX),
            Value::Short(i16::MAX),
            Value::Byte(127),
        ],
        vec![
            Value::Long(i64::MAX),
            Value::Int(i32::MAX),
            Value::Short(i16::MAX),
            Value::Byte(-128),
        ],
    ];
    check_coherence(&f, &g, &probes);

    // round-trip through the wire preserves negatives across every width.
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g);
}

#[test]
fn multi_column_lexicographic_order() {
    // column 0 most significant: order by col0, then col1, then col2.
    let f = ColumnarValueFormat::of(&[ColumnType::Int, ColumnType::Byte, ColumnType::Long]);
    let mut g: Vec<Row> = vec![
        vec![Value::Int(1), Value::Byte(5), Value::Long(100)],
        vec![Value::Int(1), Value::Byte(5), Value::Long(200)],
        vec![Value::Int(1), Value::Byte(-3), Value::Long(999)],
        vec![Value::Int(-7), Value::Byte(0), Value::Long(0)],
        vec![Value::Int(1), Value::Byte(5), Value::Long(-1)],
        vec![Value::Int(2), Value::Byte(-128), Value::Long(0)],
    ];
    g.sort_by(|a, b| f.element().compare(a, b));

    // verify the exact expected order.
    let expected: Vec<Row> = vec![
        vec![Value::Int(-7), Value::Byte(0), Value::Long(0)],
        vec![Value::Int(1), Value::Byte(-3), Value::Long(999)],
        vec![Value::Int(1), Value::Byte(5), Value::Long(-1)],
        vec![Value::Int(1), Value::Byte(5), Value::Long(100)],
        vec![Value::Int(1), Value::Byte(5), Value::Long(200)],
        vec![Value::Int(2), Value::Byte(-128), Value::Long(0)],
    ];
    assert_eq!(g, expected);

    let probes: Vec<Row> = vec![
        vec![Value::Int(-7), Value::Byte(0), Value::Long(0)], // hit
        vec![Value::Int(1), Value::Byte(5), Value::Long(150)], // miss between rows
        vec![Value::Int(1), Value::Byte(5), Value::Long(200)], // hit
        vec![Value::Int(1), Value::Byte(4), Value::Long(0)],  // miss (col1 differs)
        vec![Value::Int(3), Value::Byte(0), Value::Long(0)],  // miss past end
        vec![Value::Int(-100), Value::Byte(0), Value::Long(0)], // miss before start
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn column_cursor_reads_only_its_column() {
    // 3-column schema, row width = 8+4+2 = 14; 5 rows.
    let f = ColumnarValueFormat::of(&[ColumnType::Long, ColumnType::Int, ColumnType::Short]);
    let g: Vec<Row> = (0..5)
        .map(|i| {
            vec![
                Value::Long(1000 + i as i64),
                Value::Int(-(i as i32)),
                Value::Short((i * 7) as i16),
            ]
        })
        .collect();
    let bytes = ser(&f, &g);
    assert_eq!(bytes.len(), 5 * 14);

    // Scan column 1 (Int, width 4): expect only 5*4 = 20 payload bytes read,
    // far fewer than the whole 70-byte group.
    let mut ci = CountingInput::new(&bytes);
    let mut seen: Vec<(usize, Value)> = Vec::new();
    {
        let mut cur = f.column_cursor(&mut ci, g.len(), 1, 0, g.len()).unwrap();
        while cur.next().unwrap() {
            seen.push((cur.index(), cur.value()));
        }
    }
    assert_eq!(
        seen,
        vec![
            (0, Value::Int(0)),
            (1, Value::Int(-1)),
            (2, Value::Int(-2)),
            (3, Value::Int(-3)),
            (4, Value::Int(-4)),
        ]
    );
    assert_eq!(ci.reads, 5 * 4, "column cursor read only its column's run");
    assert!(ci.reads < bytes.len(), "did not read the whole group");
    assert_eq!(ci.pos, bytes.len(), "cursor leaves input at group end");

    // A partial range [1,4) over column 2 (Short, width 2): 3*2 = 6 bytes.
    let mut ci2 = CountingInput::new(&bytes);
    let mut seen2: Vec<(usize, Value)> = Vec::new();
    {
        let mut cur = f.column_cursor(&mut ci2, g.len(), 2, 1, 4).unwrap();
        while cur.next().unwrap() {
            seen2.push((cur.index(), cur.value()));
        }
    }
    assert_eq!(
        seen2,
        vec![
            (1, Value::Short(7)),
            (2, Value::Short(14)),
            (3, Value::Short(21)),
        ]
    );
    assert_eq!(ci2.reads, 3 * 2);
    assert_eq!(
        ci2.pos,
        bytes.len(),
        "partial column scan still lands at group end"
    );
}

#[test]
fn range_cursor_whole_row_positioning() {
    let f = ColumnarValueFormat::of(&[ColumnType::Long, ColumnType::Byte]);
    let g: Vec<Row> = vec![
        vec![Value::Long(10), Value::Byte(1)],
        vec![Value::Long(20), Value::Byte(2)],
        vec![Value::Long(30), Value::Byte(3)],
        vec![Value::Long(40), Value::Byte(4)],
        vec![Value::Long(50), Value::Byte(5)],
    ];
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp, g.len(), 1, 4).unwrap();
        let mut seen: Vec<(usize, Row)> = Vec::new();
        while cur.next().unwrap() {
            seen.push((cur.index(), cur.value()));
        }
        assert_eq!(
            seen,
            vec![
                (1, vec![Value::Long(20), Value::Byte(2)]),
                (2, vec![Value::Long(30), Value::Byte(3)]),
                (3, vec![Value::Long(40), Value::Byte(4)]),
            ]
        );
    }
    assert_eq!(
        inp.pos(),
        bytes.len(),
        "range cursor leaves input at group end"
    );

    // empty range still lands at group end.
    let mut inp2 = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp2, g.len(), 2, 2).unwrap();
        assert!(!cur.next().unwrap());
    }
    assert_eq!(inp2.pos(), bytes.len());
}

#[test]
fn empty_group_and_single_column() {
    // single BYTE column, includes an empty-group probe.
    let f = ColumnarValueFormat::of(&[ColumnType::Byte]);
    assert_eq!(f.column_count(), 1);
    assert_eq!(f.row_width(), 1);

    let empty: Vec<Row> = Vec::new();
    let bytes = ser(&f, &empty);
    assert!(bytes.is_empty());
    let mut inp = SliceInput::new(&bytes);
    // binary_search on empty group -> Err(0), leaves input at (empty) end.
    let r = f.binary_search(&vec![Value::Byte(0)], &mut inp, 0).unwrap();
    assert_eq!(r, Err(0));
    assert_eq!(inp.pos(), 0);
    assert_eq!(f.search(&empty, &vec![Value::Byte(7)]), Err(0));

    let mut g: Vec<Row> = (-3..=3).map(|v| vec![Value::Byte(v)]).collect();
    g.sort_by(|a, b| f.element().compare(a, b));
    let probes: Vec<Row> = vec![
        vec![Value::Byte(-3)],
        vec![Value::Byte(0)],
        vec![Value::Byte(3)],
        vec![Value::Byte(-128)],
        vec![Value::Byte(127)],
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn coherence_random_multicolumn() {
    // deterministic pseudo-random rows across all four widths; sort, then probe.
    let f = ColumnarValueFormat::of(&[
        ColumnType::Short,
        ColumnType::Long,
        ColumnType::Byte,
        ColumnType::Int,
    ]);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut g: Vec<Row> = (0..40)
        .map(|_| {
            vec![
                Value::Short(next() as i16),
                Value::Long(next() as i64),
                Value::Byte(next() as i8),
                Value::Int(next() as i32),
            ]
        })
        .collect();
    g.sort_by(|a, b| f.element().compare(a, b));
    g.dedup_by(|a, b| f.element().compare(a, b) == std::cmp::Ordering::Equal);

    let mut probes: Vec<Row> = g.clone(); // every stored row must be found
    for _ in 0..40 {
        probes.push(vec![
            Value::Short(next() as i16),
            Value::Long(next() as i64),
            Value::Byte(next() as i8),
            Value::Int(next() as i32),
        ]);
    }
    check_coherence(&f, &g, &probes);
}
