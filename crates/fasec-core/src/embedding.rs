//! Face embeddings: biometric templates kept out of every serialisable type (NFR-PRIV-003).

use std::fmt;

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::vision::TrackId;

/// One face embedding. Deliberately not `Serialize`/`Deserialize`; `Debug` prints only the
/// dimension, and the floats are zeroised on drop.
#[derive(Clone, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct Embedding(Box<[f32]>);

impl Embedding {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(mut v: Vec<f32>) -> Embedding {
        // `into_boxed_slice` may reallocate and free the old buffer unzeroised.
        let e = Embedding(v.as_slice().into());
        v.zeroize();
        e
    }
}

impl fmt::Debug for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Embedding(<{} floats>)", self.dim())
    }
}

/// An embedding computed for `track` on the frame stamped `ts_ms`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackEmbedding {
    pub track: TrackId,
    pub ts_ms: u64,
    pub embedding: Embedding,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_debug_is_redacted() {
        let e = Embedding::from(vec![0.123_456_7, -7.654_321, 31.4159]);
        assert_eq!(format!("{e:?}"), "Embedding(<3 floats>)");
        let te = TrackEmbedding {
            track: TrackId(7),
            ts_ms: 99,
            embedding: e,
        };
        let dbg = format!("{te:?} {te:#?}");
        for n in ["123456", "654321", "31.41"] {
            assert!(!dbg.contains(n), "{dbg}");
        }
        assert!(
            dbg.contains("TrackId(7)") && dbg.contains("<3 floats>"),
            "{dbg}"
        );
    }

    #[test]
    fn embedding_exposes_values_and_dim() {
        let e = Embedding::from(vec![1.0, 2.0]);
        assert_eq!((e.as_slice(), e.dim()), (&[1.0f32, 2.0][..], 2));
        fn zeroised_on_drop<T: ZeroizeOnDrop>() {}
        zeroised_on_drop::<Embedding>();
    }
}
