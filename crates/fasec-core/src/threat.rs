//! Threat levels.

use serde::{Deserialize, Serialize};

/// Overall threat level. Variant order is the SRS FR-PO-002 precedence, so the derived
/// `Ord` ranks `None` lowest and `Manual` highest (`max()` picks the winning threat).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLevel {
    None,
    Notice,
    OwnerAbsent,
    Onlooker,
    Spoof,
    CameraLost,
    Manual,
}

#[cfg(test)]
mod tests {
    use super::ThreatLevel::*;

    #[test]
    fn threat_order_matches_precedence() {
        let mut v = vec![
            Manual,
            Spoof,
            None,
            CameraLost,
            Onlooker,
            OwnerAbsent,
            Notice,
        ];
        v.sort();
        assert_eq!(
            v,
            vec![
                None,
                Notice,
                OwnerAbsent,
                Onlooker,
                Spoof,
                CameraLost,
                Manual
            ]
        );
    }
}
