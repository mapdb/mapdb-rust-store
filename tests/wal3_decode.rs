//! The synthetic battery for the shared WAL v3 decoder — **lesson (g)**.
//!
//! C3s's lesson (f) was "a gate built out of a comparison can only see the
//! inputs it did not compute". C3j's neighbouring one is: **a comparison can
//! only see the variation its inputs contain.** The C3j review measured three
//! decoder defects that the sample corpus cannot possibly catch, because the
//! corpus is CONSTANT in the field each one touches:
//!
//! - every sample segment's `flags` word is zero, because the writer emits the
//!   constant, so `flags = 0` hard-coded in a decoder is unfalsifiable by any
//!   bundle;
//! - every sample section's entries happen to be in ascending recid order, so
//!   a decoder that SORTED them would publish a correct-looking file — and for
//!   this port, would be graded against java's file and agree with it;
//! - the two `'K'` mark longs are both longs, so swapping them is invisible to
//!   everything downstream.
//!
//! None of the three is reachable by writing the comparison more carefully.
//! Only an input built to VARY reaches them, which is what this file is:
//! hand-built segments in shapes the corpus does not contain, framed with the
//! engine's own `pack_long` so the encoder is not a second transcription of
//! the thing under test.

#[path = "../src/store/xfix.rs"]
mod xfix;

use mapdb_rust_store::io::DataOutput2;

// ---------------------------------------------------------------------------
// builders
// ---------------------------------------------------------------------------

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

fn be64(v: i64) -> [u8; 8] {
    v.to_be_bytes()
}

/// Builds a segment the way the writer does: a 36-byte header whose CRC covers
/// the first 32 bytes, then sections whose two CRCs are taken over the header
/// bytes followed by `be64(sectionOffset)` followed by the section's own bytes.
struct SegBuilder {
    buf: Vec<u8>,
}

impl SegBuilder {
    fn new(seq: i64, first_lsn: i64, flags: u32) -> SegBuilder {
        let mut h = Vec::with_capacity(xfix::SEG_HDR);
        h.extend_from_slice(xfix::MAGIC);
        h.extend_from_slice(&be32(xfix::FORMAT_VERSION));
        h.extend_from_slice(&be32(flags));
        h.extend_from_slice(&be64(seq));
        h.extend_from_slice(&be64(first_lsn));
        h.extend_from_slice(&be32(crc32fast::hash(&h[..xfix::SEG_HDR_CRC_LEN])));
        assert_eq!(h.len(), xfix::SEG_HDR);
        SegBuilder { buf: h }
    }

    fn domain(&self, off: usize) -> crc32fast::Hasher {
        let mut c = crc32fast::Hasher::new();
        c.update(&self.buf[..xfix::SEG_HDR]);
        c.update(&(off as u64).to_be_bytes());
        c
    }

    fn push(&mut self, tag: u8, lsn: i64, body: &[u8]) -> &mut SegBuilder {
        let off = self.buf.len();
        let mut hdr = Vec::with_capacity(xfix::SEC_HDR);
        hdr.push(tag);
        hdr.extend_from_slice(&be64(lsn));
        hdr.extend_from_slice(&be64(body.len() as i64));
        let mut hc = self.domain(off);
        hc.update(&hdr[..xfix::SEC_HDR_CRC_LEN]);
        hdr.extend_from_slice(&be32(hc.finalize()));
        let mut bc = self.domain(off);
        bc.update(body);
        hdr.extend_from_slice(&be32(bc.finalize()));
        assert_eq!(hdr.len(), xfix::SEC_HDR);
        self.buf.extend_from_slice(&hdr);
        self.buf.extend_from_slice(body);
        self
    }

    fn bytes(&self) -> Vec<u8> {
        self.buf.clone()
    }
}

/// A `T_RECORD` entry. `content == None` is NULL content (`lenPlus == 0`);
/// `Some(&[])` is zero-length content (`lenPlus == 1`).
fn record(recid: u64, cap: u64, content: Option<&[u8]>) -> Vec<u8> {
    let mut o = DataOutput2::new();
    o.write_u8(xfix::T_RECORD);
    o.pack_long(recid);
    o.pack_long(cap);
    o.pack_long(content.map_or(0, |c| c.len() as u64 + 1));
    if let Some(c) = content {
        o.write_all(c);
    }
    o.copy_bytes()
}

fn tagged(tag: u8, recid: u64) -> Vec<u8> {
    let mut o = DataOutput2::new();
    o.write_u8(tag);
    o.pack_long(recid);
    o.copy_bytes()
}

fn mark_body(through: i64, log_start: i64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&be64(through));
    b.extend_from_slice(&be64(log_start));
    b
}

fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

fn one_section(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut b = SegBuilder::new(1, 1, 0);
    b.push(tag, 1, body);
    b.bytes()
}

// ---------------------------------------------------------------------------
// the three constant-field checks
// ---------------------------------------------------------------------------

/// Entries come back in WIRE order, not sorted.
///
/// Every entry stream in the sample happens to be in ascending recid order, so
/// a decoder that sorted its output would match `GOLDEN-BODY.tsv` exactly. The
/// recids here descend, which no corpus section does.
#[test]
fn entries_keep_wire_order() {
    let body = concat(&[
        record(9, 16, Some(&xfix::payload(9, 3))),
        tagged(xfix::T_DELETE, 4),
        record(2, 16, Some(&xfix::payload(2, 1))),
    ]);
    let raw = one_section(xfix::TAG_SECTION, &body);
    let seg = xfix::decode(&raw, "wire-order");
    let es = xfix::entries(&seg.sections[0], "wire-order");
    assert_eq!(
        es.iter().map(|e| e.recid).collect::<Vec<_>>(),
        vec![9, 4, 2],
        "the decoder reordered an entry stream"
    );
    assert_eq!(
        es.iter().map(|e| e.kind()).collect::<Vec<_>>(),
        vec!["RECORD", "DELETE", "RECORD"]
    );
}

/// NULL content and zero-length content stay distinct, in both columns.
///
/// `lenPlus == 0` is NULL and `lenPlus == 1` is a zero-length record. A decoder
/// that turned `lenPlus` into a length would report `0` for both, and two
/// engines that both did it would agree forever.
#[test]
fn null_and_zero_length_records_are_distinct() {
    let body = concat(&[record(1, 0, None), record(2, 16, Some(&[]))]);
    let raw = one_section(xfix::TAG_SECTION, &body);
    let seg = xfix::decode(&raw, "null-vs-empty");
    let es = xfix::entries(&seg.sections[0], "null-vs-empty");

    assert_eq!(
        es[0].len_plus,
        Some(0),
        "NULL content must decode to lenPlus 0"
    );
    assert_eq!(
        es[0].content, None,
        "a NULL record carries no content bytes"
    );
    assert_eq!(es[0].cap, Some(0), "a NULL record's cap is 0");

    assert_eq!(
        es[1].len_plus,
        Some(1),
        "zero-length content must decode to lenPlus 1"
    );
    assert_eq!(
        es[1].content,
        Some(Vec::new()),
        "a zero-length record carries an empty content slice, not None"
    );
    assert_ne!(es[0].content, es[1].content);
}

/// The two `'K'` mark longs come back in wire order.
///
/// Both fields are longs in one 16-byte body, so a swap is invisible to every
/// consumer downstream of the decoder. `(3, 4)` is asymmetric and the segment's
/// own sequence is 7, so the swap is detectable here and only here.
#[test]
fn mark_fields_are_in_order() {
    let mut b = SegBuilder::new(7, 4, 0);
    b.push(xfix::TAG_MARK, 4, &mark_body(3, 4));
    let raw = b.bytes();
    let seg = xfix::decode(&raw, "mark-order");
    assert_eq!(
        xfix::mark(&seg.sections[0], "mark-order"),
        (3, 4),
        "the two mark longs came back swapped"
    );
}

/// Every header field is actually read out of the bytes.
///
/// `flags` is the one the corpus cannot check at all: the writer emits the
/// constant 0, so `flags = 0` hard-coded in a decoder matches every bundle
/// that exists. `seq` and `first_lsn` get byte patterns that would survive a
/// truncation to 32 bits looking wrong.
#[test]
fn header_fields_are_decoded() {
    let raw = {
        let mut b = SegBuilder::new(0x0102_0304_0506_0708, 0x1112_1314_1516_1718, 0x2A);
        b.push(
            xfix::TAG_SECTION,
            1,
            &record(1, 16, Some(&xfix::payload(1, 2))),
        );
        b.bytes()
    };
    let seg = xfix::decode(&raw, "header");
    assert_eq!(seg.header.version, xfix::FORMAT_VERSION);
    assert_eq!(seg.header.flags, 0x2A, "the flags word is not decoded");
    assert_eq!(seg.header.seq, 0x0102_0304_0506_0708);
    assert_eq!(seg.header.first_lsn, 0x1112_1314_1516_1718);
    assert_eq!(
        seg.header.header_crc,
        crc32fast::hash(&raw[..xfix::SEG_HDR_CRC_LEN])
    );
}

// ---------------------------------------------------------------------------
// framing
// ---------------------------------------------------------------------------

/// Section offsets advance by exactly one header plus one body, and a section
/// is not relocatable: the offset is inside its CRC domain.
#[test]
fn section_offsets_advance_and_bind_the_section() {
    let s1 = record(1, 16, Some(&xfix::payload(1, 5)));
    let s2 = record(2, 16, Some(&xfix::payload(2, 40)));
    let mut b = SegBuilder::new(1, 1, 0);
    b.push(xfix::TAG_SECTION, 1, &s1);
    b.push(xfix::TAG_SECTION, 2, &s2);
    let raw = b.bytes();

    let seg = xfix::decode(&raw, "offsets");
    assert_eq!(seg.sections.len(), 2);
    assert_eq!(seg.sections[0].offset, xfix::SEG_HDR);
    assert_eq!(
        seg.sections[1].offset,
        xfix::SEG_HDR + xfix::SEC_HDR + s1.len()
    );
    assert_eq!(seg.sections[0].index, 0);
    assert_eq!(seg.sections[1].index, 1);
    assert_eq!(seg.trailing, 0);

    // The same section bytes at a different offset must fail: the offset is in
    // the CRC domain, which is what makes a section un-relocatable. Sixteen
    // filler bytes are spliced in ahead of the second section, moving it.
    let cut = seg.sections[1].offset;
    let mut moved = raw[..cut].to_vec();
    moved.extend_from_slice(&[0u8; 16]);
    moved.extend_from_slice(&raw[cut..]);
    xfix::assert_refused("a section whose bytes were moved to another offset", || {
        xfix::decode(&moved, "moved");
    });
}

/// A damaged segment is refused, one damage at a time.
#[test]
fn a_damaged_segment_is_refused() {
    let good = {
        let mut b = SegBuilder::new(1, 1, 0);
        b.push(
            xfix::TAG_SECTION,
            1,
            &record(1, 16, Some(&xfix::payload(1, 9))),
        );
        b.bytes()
    };
    xfix::decode(&good, "control"); // the control: undamaged, it decodes

    /// One named single-byte-scale damage applied to a copy of the control.
    type Damage = Box<dyn Fn(&mut Vec<u8>)>;
    let cases: Vec<(&str, Damage)> = vec![
        (
            "a file shorter than a segment header",
            Box::new(|r: &mut Vec<u8>| r.truncate(35)),
        ),
        ("bad magic", Box::new(|r: &mut Vec<u8>| r[0] ^= 0xFF)),
        (
            "a future format version",
            Box::new(|r: &mut Vec<u8>| r[11] = 4),
        ),
        (
            "a wrong header CRC",
            Box::new(|r: &mut Vec<u8>| r[32] ^= 0xFF),
        ),
        (
            "an unknown section tag",
            Box::new(|r: &mut Vec<u8>| r[36] = b'Z'),
        ),
        (
            "a wrong section-header CRC",
            Box::new(|r: &mut Vec<u8>| r[36 + 17] ^= 0xFF),
        ),
        (
            "a wrong section-body CRC",
            Box::new(|r: &mut Vec<u8>| r[36 + 21] ^= 0xFF),
        ),
        (
            "a flipped content byte",
            Box::new(|r: &mut Vec<u8>| {
                let n = r.len();
                r[n - 1] ^= 0xFF;
            }),
        ),
    ];
    for (what, damage) in cases {
        let mut raw = good.clone();
        damage(&mut raw);
        assert_ne!(
            raw, good,
            "the {what} case did not actually change anything"
        );
        xfix::assert_refused(what, move || {
            xfix::decode(&raw, "damaged");
        });
    }
}

/// A torn tail is REPORTED, not silently dropped and not refused.
///
/// A writer that died mid-append leaves a partial section, and recovery's whole
/// job is to stop at the last complete one. The decoder therefore hands back
/// what it could read plus a byte count, and the caller decides — the golden
/// comparisons require `trailing == 0`, because a published fixture with a torn
/// tail would be a different bug.
#[test]
fn a_torn_tail_is_reported() {
    let mut b = SegBuilder::new(1, 1, 0);
    b.push(
        xfix::TAG_SECTION,
        1,
        &record(1, 16, Some(&xfix::payload(1, 9))),
    );
    b.push(
        xfix::TAG_SECTION,
        2,
        &record(2, 16, Some(&xfix::payload(2, 9))),
    );
    let raw = b.bytes();

    // (a) a body that is not all there
    let torn_body = raw[..raw.len() - 4].to_vec();
    let seg = xfix::decode(&torn_body, "torn-body");
    assert_eq!(seg.sections.len(), 1, "the complete prefix is one section");
    assert_eq!(
        seg.trailing,
        torn_body.len()
            - seg.sections[0].offset
            - xfix::SEC_HDR
            - seg.sections[0].body_len as usize
    );
    assert!(seg.trailing > 0);

    // (b) not even a whole section header
    let torn_hdr = raw[..raw.len() - 30].to_vec();
    let seg = xfix::decode(&torn_hdr, "torn-header");
    assert_eq!(seg.sections.len(), 1);
    assert!(seg.trailing > 0 && seg.trailing < xfix::SEC_HDR);
}

// ---------------------------------------------------------------------------
// entry streams
// ---------------------------------------------------------------------------

/// An entry stream that ends mid-value, or names a tag the dump has no columns
/// for, is refused rather than truncated into a plausible list.
#[test]
fn a_malformed_entry_stream_is_refused() {
    let full = record(1, 16, Some(&xfix::payload(1, 40)));
    for cut in [1usize, 2, 4, full.len() - 1] {
        let raw = one_section(xfix::TAG_SECTION, &full[..cut]);
        xfix::assert_refused(&format!("an entry stream cut to {cut} bytes"), move || {
            let seg = xfix::decode(&raw, "cut");
            xfix::entries(&seg.sections[0], "cut");
        });
    }

    let unknown = one_section(xfix::TAG_SECTION, &tagged(9, 1));
    xfix::assert_refused("an unknown entry tag", move || {
        let seg = xfix::decode(&unknown, "unknown-tag");
        xfix::entries(&seg.sections[0], "unknown-tag");
    });

    // T_APPEND is a real engine op that no fixture exercises and the body dump
    // has no columns for. Refusing by name is the honest answer: decoding it
    // into a row shape that was never designed would be a silent guess.
    let append = one_section(xfix::TAG_SECTION, &tagged(xfix::T_APPEND, 1));
    xfix::assert_refused("a T_APPEND entry", move || {
        let seg = xfix::decode(&append, "append");
        xfix::entries(&seg.sections[0], "append");
    });

    // A 'K' body is a mark, not an entry stream, and must not be decoded as one.
    let mut b = SegBuilder::new(7, 4, 0);
    b.push(xfix::TAG_MARK, 4, &mark_body(3, 4));
    let k = b.bytes();
    xfix::assert_refused("a 'K' body read as an entry stream", move || {
        let seg = xfix::decode(&k, "k-as-entries");
        xfix::entries(&seg.sections[0], "k-as-entries");
    });
}

/// `'C'` sections carry an ordinary entry stream.
///
/// The two tags differ in what recovery DOES with the section, not in how a
/// body is framed (`StoreWAL.java:850`), so a decoder that special-cased `'C'`
/// would silently drop the cleaner image — which is the entire content of the
/// `wal3-java-cleaned` bundle.
#[test]
fn image_sections_decode_like_ordinary_ones() {
    let body = concat(&[
        record(1, 16, Some(&xfix::payload(1, 4))),
        tagged(xfix::T_PREALLOC, 3),
    ]);
    let s = one_section(xfix::TAG_SECTION, &body);
    let c = one_section(xfix::TAG_IMAGE, &body);
    let from_s = xfix::entries(&xfix::decode(&s, "s").sections[0], "s");
    let from_c = xfix::entries(&xfix::decode(&c, "c").sections[0], "c");
    assert_eq!(from_s, from_c, "'C' and 'S' bodies must decode identically");
}

// ---------------------------------------------------------------------------
// the two independent witnesses
// ---------------------------------------------------------------------------

/// `check_cap` is the only thing in the slice that observes `cap` independently.
///
/// Replay consumes `cap` and exposes only the resulting record, and the golden
/// comparison grades the column against a file another engine wrote — so an
/// emitter that consumed the varint correctly and then printed a fabricated
/// number would pass both. The C3j review named exactly that hole.
#[test]
fn the_cap_witness_accepts_only_real_capacities() {
    xfix::check_cap(0, 1_000_000, "linked/oversize content is cap 0");
    xfix::check_cap(16, 0, "the smallest plain capacity");
    xfix::check_cap(128, 121, "4 + 121 = 125, rounded up to 128");

    for (cap, len, what) in [
        (127i64, 121usize, "a capacity that is not 16-aligned"),
        (120, 121, "a capacity with no room for the content"),
        (124, 120, "a capacity with no room for the 4-byte header"),
        (-16, 0, "a negative capacity"),
    ] {
        xfix::assert_refused(what, move || xfix::check_cap(cap, len, what));
    }
}

/// `check_mark` is the only thing that observes the mark longs' MEANING.
///
/// Both fields are longs, so the decoder cannot tell a swap from a legal pair
/// on its own. These are the engine's own S8/K4 rules, restated over what the
/// dump can see, and they are what makes [`mark_fields_are_in_order`] more
/// than a tautology on the real corpus: the sample's `(2, 9)` in segment 4
/// becomes `(9, 2)` under a swap, and K4 rejects it.
#[test]
fn the_mark_witness_enforces_s8_and_k4() {
    xfix::check_mark((2, 9), 4, 10, "the sample's own mark");

    for (m, seq, lsn, what) in [
        (
            (9i64, 2i64),
            4i64,
            10i64,
            "the sample's mark with its two longs swapped",
        ),
        ((0, 9), 4, 10, "a cleanedThroughSeq of zero"),
        (
            (4, 9),
            4,
            10,
            "a mark authorizing the removal of its own segment (K4)",
        ),
        (
            (5, 9),
            4,
            10,
            "a mark authorizing the removal of a later segment (K4)",
        ),
        ((2, 0), 4, 10, "a logStartLsn of zero (S8)"),
        (
            (2, 11),
            4,
            10,
            "a logStartLsn beyond the mark's own LSN (S8)",
        ),
    ] {
        xfix::assert_refused(what, move || xfix::check_mark(m, seq, lsn, what));
    }
}
