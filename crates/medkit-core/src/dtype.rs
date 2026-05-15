/// Scalar data types commonly encountered in medical image storage and training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// Boolean values.
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// 16-bit floating point.
    F16,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
}

impl DType {
    /// Returns the number of bytes per scalar value.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Returns true for floating point data types.
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// Returns true for integer data types.
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::U8 | Self::I8 | Self::U16 | Self::I16 | Self::U32 | Self::I32
        )
    }

    /// Returns true for signed numeric data types.
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::F16 | Self::F32 | Self::F64
        )
    }
}
