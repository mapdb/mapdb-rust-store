//! Heterogeneous row/cell value (decision D3). Java `Object[]` of boxed
//! primitives / String / byte[] → a closed enum with exhaustive match and Ord.
//! Used by TupleFormat rows and ColumnarValueFormat cells.

/// A single tuple component or columnar cell.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    Long(i64),
    Int(i32),
    Short(i16),
    Byte(i8),
    Str(String),
    Bytes(Vec<u8>),
}

impl Value {
    pub fn as_long(&self) -> Option<i64> {
        match self {
            Value::Long(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i32> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }
}
