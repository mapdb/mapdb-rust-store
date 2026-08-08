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

/// Builds a segment the way the writer does — and, unlike a byte-poking
/// mutator, RESEALS everything from the fields it is given.
///
/// That property is the point. The CRC domains chain: the segment header CRC
/// covers the first 32 bytes, so corrupting the magic or the version breaks it
/// too; and every section CRC is taken over ALL 36 header bytes followed by
/// `be64(sectionOffset)`, so corrupting any header byte — including the stored
/// header CRC itself — breaks every section CRC as well. A test that flips a
/// magic byte in a finished segment therefore does not test the magic check:
/// deleting that check outright still leaves the input refused, by a CRC. The
/// C3r review measured exactly that on four checks here. Building from fields
/// and resealing is what isolates them.
struct SegBuilder {
    magic: [u8; 8],
    version: u32,
    flags: u32,
    seq: i64,
    first_lsn: i64,
    /// Replaces the computed header CRC, to test the header-CRC check ALONE:
    /// the sections are then sealed against these header bytes, so they stay
    /// valid and only the header check can fire.
    header_crc: Option<u32>,
    sections: Vec<(u8, i64, Vec<u8>)>,
}

impl SegBuilder {
    fn new(seq: i64, first_lsn: i64, flags: u32) -> SegBuilder {
        SegBuilder {
            magic: *xfix::MAGIC,
            version: xfix::FORMAT_VERSION,
            flags,
            seq,
            first_lsn,
            header_crc: None,
            sections: Vec::new(),
        }
    }

    fn header(&self) -> Vec<u8> {
        let mut h = Vec::with_capacity(xfix::SEG_HDR);
        h.extend_from_slice(&self.magic);
        h.extend_from_slice(&be32(self.version));
        h.extend_from_slice(&be32(self.flags));
        h.extend_from_slice(&be64(self.seq));
        h.extend_from_slice(&be64(self.first_lsn));
        let crc = self
            .header_crc
            .unwrap_or_else(|| crc32fast::hash(&h[..xfix::SEG_HDR_CRC_LEN]));
        h.extend_from_slice(&be32(crc));
        assert_eq!(h.len(), xfix::SEG_HDR);
        h
    }

    fn push(&mut self, tag: u8, lsn: i64, body: &[u8]) -> &mut SegBuilder {
        self.sections.push((tag, lsn, body.to_vec()));
        self
    }

    fn bytes(&self) -> Vec<u8> {
        let head = self.header();
        let mut out = head.clone();
        for (tag, lsn, body) in &self.sections {
            let off = out.len();
            let domain = |extra: &[u8]| {
                let mut c = crc32fast::Hasher::new();
                c.update(&head);
                c.update(&(off as u64).to_be_bytes());
                c.update(extra);
                c.finalize()
            };
            let mut hdr = Vec::with_capacity(xfix::SEC_HDR);
            hdr.push(*tag);
            hdr.extend_from_slice(&be64(*lsn));
            hdr.extend_from_slice(&be64(body.len() as i64));
            let hc = domain(&hdr[..xfix::SEC_HDR_CRC_LEN]);
            hdr.extend_from_slice(&be32(hc));
            hdr.extend_from_slice(&be32(domain(body)));
            assert_eq!(hdr.len(), xfix::SEC_HDR);
            out.extend_from_slice(&hdr);
            out.extend_from_slice(body);
        }
        out
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

/// Each header/framing check is refused **on an input only that check rejects**.
///
/// The C3r review found the previous version of this test unable to fail: it
/// poked bytes in a finished segment, and because the CRC domains chain, a
/// flipped magic byte is also a broken header CRC and a broken section CRC.
/// Deleting the magic check left the input refused anyway. So the cases below
/// are BUILT, not poked: every one is a fully-sealed segment that differs from
/// the control in exactly one semantic field, and the only rule that can refuse
/// it is the rule it is named for.
#[test]
fn each_header_check_is_refused_on_an_input_only_it_rejects() {
    let entry = record(1, 16, Some(&xfix::payload(1, 9)));
    let control = {
        let mut b = SegBuilder::new(1, 1, 0);
        b.push(xfix::TAG_SECTION, 1, &entry);
        b.bytes()
    };
    xfix::decode(&control, "control");

    // (a) bad magic, everything else sealed around it.
    let mut b = SegBuilder::new(1, 1, 0);
    b.magic = *b"MDBS.XXX";
    b.push(xfix::TAG_SECTION, 1, &entry);
    let raw = b.bytes();
    assert_eq!(
        crc32fast::hash(&raw[..xfix::SEG_HDR_CRC_LEN]),
        u32::from_be_bytes(raw[32..36].try_into().unwrap()),
        "the bad-magic case must still carry a VALID header CRC, or it does not isolate"
    );
    xfix::assert_refused("bad magic, with every CRC valid", move || {
        xfix::decode(&raw, "bad-magic");
    });

    // (b) a future format version, everything else sealed around it.
    let mut b = SegBuilder::new(1, 1, 0);
    b.version = xfix::FORMAT_VERSION + 1;
    b.push(xfix::TAG_SECTION, 1, &entry);
    let raw = b.bytes();
    xfix::assert_refused("a future format version, with every CRC valid", move || {
        xfix::decode(&raw, "future-version");
    });

    // (c) a wrong header CRC — and the sections resealed against the header
    // bytes that carry it, so the section CRCs are valid and only the header
    // check can fire.
    let mut b = SegBuilder::new(1, 1, 0);
    b.header_crc = Some(0xDEAD_BEEF);
    b.push(xfix::TAG_SECTION, 1, &entry);
    let raw = b.bytes();
    xfix::assert_refused("a wrong header CRC, with valid section CRCs", move || {
        xfix::decode(&raw, "bad-header-crc");
    });

    // (d) an unknown section tag, with its own section-header CRC recomputed
    // over the new tag. Poking the tag byte alone would break that CRC and be
    // refused by it instead.
    let mut b = SegBuilder::new(1, 1, 0);
    b.push(b'Z', 1, &entry);
    let raw = b.bytes();
    xfix::assert_refused("an unknown section tag, correctly sealed", move || {
        xfix::decode(&raw, "bad-tag");
    });

    // (e) a 'K' body that is not 16 bytes, correctly sealed.
    let mut b = SegBuilder::new(7, 4, 0);
    b.push(xfix::TAG_MARK, 4, &[0u8; 8]);
    let raw = b.bytes();
    xfix::assert_refused("a 'K' section whose body is not 16 bytes", move || {
        xfix::decode(&raw, "short-mark");
    });

    // (f) shorter than a segment header at all.
    let raw = control[..35].to_vec();
    xfix::assert_refused("a file shorter than a segment header", move || {
        xfix::decode(&raw, "short");
    });
}

/// The two stored CRCs are checked, on inputs where nothing else has changed.
///
/// These two ARE isolated by byte-poking, and that is not an accident: each
/// stored CRC field sits OUTSIDE its own domain (the section-header CRC covers
/// bytes 0..17 of the header, the body CRC covers the body), so overwriting one
/// invalidates that check and no other.
#[test]
fn the_two_section_crcs_are_checked() {
    let good = {
        let mut b = SegBuilder::new(1, 1, 0);
        b.push(
            xfix::TAG_SECTION,
            1,
            &record(1, 16, Some(&xfix::payload(1, 9))),
        );
        b.bytes()
    };
    for (what, at) in [
        ("a wrong section-header CRC", xfix::SEG_HDR + 17),
        ("a wrong section-body CRC", xfix::SEG_HDR + 21),
    ] {
        let mut raw = good.clone();
        raw[at] ^= 0xFF;
        xfix::assert_refused(what, move || {
            xfix::decode(&raw, "damaged");
        });
    }
    // ...and any change to the body itself is caught by the body CRC.
    let mut raw = good.clone();
    let n = raw.len();
    raw[n - 1] ^= 0xFF;
    xfix::assert_refused("a flipped content byte", move || {
        xfix::decode(&raw, "flipped");
    });
}

/// An INCOMPLETE FINAL SECTION is reported, not refused.
///
/// **This is not the engine's torn-tail policy, and the name says so on
/// purpose.** `wal_recover` decides tornness with context this decoder does not
/// have: it also treats a damaged final section header or body CRC as a torn
/// active tail when no valid later section proves mid-log corruption, and it
/// treats an overrunning body BELOW the highest segment as corruption rather
/// than tornness. This helper has no segment-position context and refuses on any
/// CRC failure, so the two disagree in both directions on inputs the published
/// fixtures do not contain. That is a deliberate scope choice for a
/// comparison-only decoder — every pinned file is required to have
/// `trailing == 0` and is separately opened by the real engine — and the C3r
/// review was right that the earlier wording claimed more than this.
///
/// What IS covered: the two shapes where framing simply runs out.
#[test]
fn an_incomplete_final_section_is_reported_not_refused() {
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

    // Truncated T_APPEND (tag+recid only) fails mid-delta.
    let append_trunc = one_section(xfix::TAG_SECTION, &tagged(xfix::T_APPEND, 1));
    xfix::assert_refused("a truncated T_APPEND entry", move || {
        let seg = xfix::decode(&append_trunc, "append-trunc");
        xfix::entries(&seg.sections[0], "append-trunc");
    });

    // delta must be in [1, lsn-1]; at section LSN 1 no legal delta exists.
    let mut bad = DataOutput2::new();
    bad.write_u8(xfix::T_APPEND);
    bad.pack_long(1);
    bad.pack_long(1);
    bad.pack_long(0);
    let append_bad_delta = one_section(xfix::TAG_SECTION, &bad.copy_bytes());
    xfix::assert_refused("a T_APPEND with delta outside [1, lsn-1]", move || {
        let seg = xfix::decode(&append_bad_delta, "append-bad-delta");
        xfix::entries(&seg.sections[0], "append-bad-delta");
    });

    // Legal delta, overlong len: claims more payload than remains (C9a §4.3).
    let mut over = DataOutput2::new();
    over.write_u8(xfix::T_APPEND);
    over.pack_long(1);
    over.pack_long(1);
    over.pack_long(100);
    let mut b_over = SegBuilder::new(1, 5, 0);
    b_over.push(xfix::TAG_SECTION, 5, &over.copy_bytes());
    let append_over_len = b_over.bytes();
    xfix::assert_refused("a T_APPEND whose len overruns the section body", move || {
        let seg = xfix::decode(&append_over_len, "append-over-len");
        xfix::entries(&seg.sections[0], "append-over-len");
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

/// Well-formed T_APPEND decodes the four O1 fields (C9a).
#[test]
fn append_entries_decode_four_fields() {
    let mut o = DataOutput2::new();
    o.write_u8(xfix::T_APPEND);
    o.pack_long(7);
    o.pack_long(1); // delta
    o.pack_long(3); // len
    o.write_all(&[10, 20, 30]);
    let mut b = SegBuilder::new(1, 5, 0);
    b.push(xfix::TAG_SECTION, 5, &o.copy_bytes());
    let raw = b.bytes();
    let es = xfix::entries(&xfix::decode(&raw, "append").sections[0], "append");
    assert_eq!(es.len(), 1);
    let e = &es[0];
    assert_eq!(e.kind(), "APPEND");
    assert_eq!(e.recid, 7);
    assert_eq!(e.delta, Some(1));
    assert_eq!(e.base_lsn, Some(4));
    assert_eq!(e.append_len, Some(3));
    assert_eq!(e.content.as_deref(), Some(&[10, 20, 30][..]));
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
    // The plain-record ceiling, restated here so the boundary cases below read
    // as boundaries. `xfix_ro` checks it against the engine's own constant.
    const MAX: i64 = xfix::MAX_CAPACITY;
    assert_eq!(MAX, 1_048_528);

    xfix::check_cap(16, 0, "the smallest plain capacity");
    xfix::check_cap(128, 121, "4 + 121 = 125, rounded up to 128");
    xfix::check_cap(MAX, MAX as usize - 4, "the largest plain record");
    // cap 0 means the content was too big for a plain record and went linked.
    // The smallest length for which that is true is the one where 4 + len first
    // exceeds the ceiling.
    xfix::check_cap(
        0,
        MAX as usize - 3,
        "the smallest genuinely oversize record",
    );

    for (cap, len, what) in [
        // The case that made the whole witness weaker than the engine: cap 0
        // claims the content was stored linked, but 1_000_004 fits a plain
        // record, so recovery rejects precisely what this used to bless.
        (
            0i64,
            1_000_000usize,
            "a zero cap on content that fits a plain record",
        ),
        (
            0,
            MAX as usize - 4,
            "a zero cap on content that exactly fills the ceiling",
        ),
        (MAX + 16, 0, "a capacity above the plain-record ceiling"),
        (127, 121, "a capacity that is not 16-aligned"),
        (
            112,
            121,
            "a 16-aligned capacity with no room for the content",
        ),
        (
            128,
            125,
            "a 16-aligned capacity with no room for the 4-byte header",
        ),
        (-16, 0, "a negative capacity"),
    ] {
        xfix::assert_refused(what, move || xfix::check_cap(cap, len, what));
    }
}

/// `check_payload` rejects bytes OUTSIDE the payload function.
///
/// It is a language-membership test, not a corpus-membership test, and the
/// distinction is the C3r review's: it accepts any `(id, len)` pair without
/// consulting a fixture's history, so it proves these bytes are *a* payload,
/// not that this bundle issued *this* payload. What it is for is the one thing
/// nothing else here reaches — the engine's replay never shows content bytes to
/// the suite, and the golden sha column grades them against a file another
/// engine wrote — so a decoder that landed on the wrong offset would otherwise
/// be checked only against itself.
#[test]
fn the_payload_witness_rejects_bytes_the_corpus_never_issued() {
    xfix::check_payload(&xfix::payload(0, 32), "id 0");
    xfix::check_payload(&xfix::payload(255, 700), "the largest id");
    xfix::check_payload(&[], "zero-length content carries no id");

    let mut off_by_one = xfix::payload(103, 120);
    off_by_one[7] ^= 0x01;
    xfix::assert_refused("a single corrupted content byte", move || {
        xfix::check_payload(&off_by_one, "corrupted");
    });

    // The shape a mis-framed entry stream actually produces: a run that begins
    // on the entry's varint bytes and only then reaches the payload.
    let mut framed = vec![0x81u8, 0x90, 0x02];
    framed.extend(xfix::payload(103, 20));
    xfix::assert_refused(
        "content read starting on an entry's varint bytes",
        move || {
            xfix::check_payload(&framed, "framed");
        },
    );

    // ...and a run that spans the boundary between two records' payloads.
    let mut spliced = xfix::payload(50, 10);
    spliced.extend(xfix::payload(60, 10));
    xfix::assert_refused("content spanning two records' payloads", move || {
        xfix::check_payload(&spliced, "spliced");
    });

    // WHAT THIS WITNESS DOES NOT CATCH, measured rather than assumed.
    // `payload` is an arithmetic progression in i, so EVERY suffix of a payload
    // is itself a payload under a different id: payload(id, n)[k..] ==
    // payload((id + 131k) & 0xff, n - k). A decode shifted by k bytes WITHIN one
    // record's content is therefore invisible here — it is caught instead by the
    // `lenPlus` length check next to the call, and by the sha column of
    // GOLDEN-BODY.tsv. This is a property of the corpus's payload function, so
    // it is the same in all three ports; stating it is cheaper than each of them
    // rediscovering it.
    let k = 3usize;
    assert_eq!(
        xfix::payload(103, 120)[k..],
        xfix::payload((103 + 131 * k as u64) & 0xff, 120 - k)[..]
    );
}

/// The recid cross-check is ONE-WAY, and it fires.
///
/// Every recid the manifest names must be witnessed in the decoded history;
/// never the reverse. Plan §5 forbids the reverse — a rolled-back put need only
/// be invisible through the API, and `wal3-java-tail` already carries recids
/// beyond the six §5.2 describes — so set equality would be a violation waiting
/// for the first legal fixture. Both halves are asserted here: the surplus
/// direction must be TOLERATED, the missing direction must be REFUSED.
#[test]
fn the_recid_cross_check_is_one_way_and_can_fail() {
    let text = "version\t2\nfixture\tf\twal3-namespace\tjava\tc\n\
                file\tf\tx.wal.0000000000000001\t36\taa\tbb\n\
                recid\tf\tr1\t1\tlive\t1\t8\n\
                recid\tf\tr2\t2\tnull\t0\t0\n";
    let loaded = xfix::parse(text);
    let m = loaded.v2();

    // exactly the named recids, and a superset: both fine.
    xfix::check_recids_against_manifest(m, "f", &[1u64, 2].into_iter().collect());
    xfix::check_recids_against_manifest(m, "f", &[1u64, 2, 7, 99].into_iter().collect());

    for (seen, what) in [
        (vec![1u64], "a decode that never mentions recid 2"),
        (vec![], "a decode that mentions no recid at all"),
        (vec![3u64, 4], "a decode whose recids are all shifted"),
    ] {
        let set: std::collections::BTreeSet<u64> = seen.into_iter().collect();
        xfix::assert_refused(what, || {
            xfix::check_recids_against_manifest(m, "f", &set);
        });
    }

    xfix::assert_refused("a fixture with no recid rows to check against", || {
        xfix::check_recids_against_manifest(m, "nonexistent", &Default::default());
    });
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
