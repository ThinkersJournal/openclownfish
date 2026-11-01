//! Shared by the device differentials: the narrow integer dtypes' test data
//! and element widths.

use unpopped::oracle::TypedBuffer;
use unpopped_vocab::ElementKind;

/// The four narrow integer dtypes.
pub const NARROW: [ElementKind; 4] = [
    ElementKind::I8,
    ElementKind::U8,
    ElementKind::I16,
    ElementKind::U16,
];

/// Bytes per element.
pub fn width(d: ElementKind) -> usize {
    match d {
        ElementKind::I8 | ElementKind::U8 => 1,
        ElementKind::I16 | ElementKind::U16 => 2,
        ElementKind::I64 | ElementKind::F64 => 8,
        _ => 4,
    }
}

/// The values where wrapping and sign go wrong at each width: both extremes,
/// -1 (or the largest half), 0-adjacent small values, and values whose sums and
/// products overflow the element type.
pub fn pool(d: ElementKind) -> [i64; 9] {
    match d {
        ElementKind::I8 => [127, 1, -1, 2, 7, -7, -128, 3, 100],
        ElementKind::U8 => [255, 1, 2, 7, 200, 128, 3, 100, 127],
        ElementKind::I16 => [32767, 1, -1, 2, 7, -7, -32768, 3, 12345],
        ElementKind::U16 => [65535, 1, 2, 7, 40000, 32768, 3, 100, 12345],
        other => panic!("not a narrow dtype: {other:?}"),
    }
}

/// `shape`'s worth of `d` data cycling through [`pool`], or `None` for a dtype
/// that is not narrow.
#[allow(dead_code)]
pub fn narrow_buffer(d: ElementKind, shape: &[i64]) -> Option<TypedBuffer> {
    if !NARROW.contains(&d) {
        return None;
    }
    let n: usize = shape.iter().product::<i64>() as usize;
    let p = pool(d);
    let v = |i: usize| p[i % p.len()];
    Some(match d {
        ElementKind::I8 => {
            let x: Vec<i8> = (0..n).map(|i| v(i) as i8).collect();
            TypedBuffer::from_i8(shape, &x)
        }
        ElementKind::U8 => {
            let x: Vec<u8> = (0..n).map(|i| v(i) as u8).collect();
            TypedBuffer::from_u8(shape, &x)
        }
        ElementKind::I16 => {
            let x: Vec<i16> = (0..n).map(|i| v(i) as i16).collect();
            TypedBuffer::from_i16(shape, &x)
        }
        _ => {
            let x: Vec<u16> = (0..n).map(|i| v(i) as u16).collect();
            TypedBuffer::from_u16(shape, &x)
        }
    })
}

/// A narrow-dtype buffer of the given values (each already inside the type's range).
#[allow(dead_code)]
pub fn narrow_from(d: ElementKind, shape: &[i64], vals: &[i64]) -> TypedBuffer {
    match d {
        ElementKind::I8 => {
            let x: Vec<i8> = vals.iter().map(|&v| v as i8).collect();
            TypedBuffer::from_i8(shape, &x)
        }
        ElementKind::U8 => {
            let x: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
            TypedBuffer::from_u8(shape, &x)
        }
        ElementKind::I16 => {
            let x: Vec<i16> = vals.iter().map(|&v| v as i16).collect();
            TypedBuffer::from_i16(shape, &x)
        }
        ElementKind::U16 => {
            let x: Vec<u16> = vals.iter().map(|&v| v as u16).collect();
            TypedBuffer::from_u16(shape, &x)
        }
        other => panic!("not a narrow dtype: {other:?}"),
    }
}

/// The little-endian bytes of `vals` at `d`'s width.
#[allow(dead_code)]
pub fn narrow_bytes(d: ElementKind, vals: &[i64]) -> Vec<u8> {
    let w = width(d);
    vals.iter()
        .flat_map(|v| v.to_le_bytes()[..w].to_vec())
        .collect()
}
