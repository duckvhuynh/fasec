//! The replay tool's shared half: the local paired RGB + IR recording format (written by the
//! `ir_debug` example, read by `fasec-replay run --ir-input`) and the IR cue probe both use.

pub mod probe;
pub mod recording;
