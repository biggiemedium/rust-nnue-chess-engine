//! The quantized network, loaded from the file training/nnue/export.py writes.
//!
//! Architecture: (768 -> 256) x 2 perspectives -> SCReLU -> 1
//!
//! File layout (little endian), see docs/training.md:
//!   magic       b"NNUE"
//!   version     u32
//!   hidden      u32
//!   ft_weights  i16[768 * hidden]   (input-major: row i = weights for feature i)
//!   ft_bias     i16[hidden]
//!   out_weights i16[2 * hidden]     (side to move half first, then opponent)
//!   out_bias    i32
//!
//! https://www.chessprogramming.org/NNUE

pub const HIDDEN: usize = 256;

// quantization scales, these MUST match training/nnue/export.py
pub const QA: i32 = 255; // feature transformer (accumulator)
pub const QB: i32 = 64; // output layer
pub const EVAL_SCALE: i32 = 400; // net output 1.0 == 400 centipawns

// TODO: Network struct + load(path)
// TODO: evaluate(stm_acc, nstm_acc) -> i32
//   SCReLU in ints: clamp(x, 0, QA)^2
//   cp = (sum(screlu(acc) * w) / QA + bias) * EVAL_SCALE / (QA * QB)
