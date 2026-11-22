//! Enrolment flows, face gallery and encrypted template store.

pub mod enrol;
pub mod error;
pub mod matcher;
mod pin;
pub mod protect;
pub mod store;
pub mod tracker;
pub mod voter;

pub use enrol::{
    EnrolConfig, EnrolError, EnrolledSample, Enrolment, PoseWindow, RejectReason, SampleVerdict,
    sharpness,
};
pub use error::IdentityError;
pub use matcher::{MatchResult, Matcher, MatcherConfig};
pub use protect::{ProtectError, SecretProtector};
pub use store::{
    DecryptedTemplate, EMBEDDING_DIM, Gallery, Person, PersonId, Pose, Role, StoredTemplate,
};
pub use tracker::IdentityTracker;
pub use voter::{Voter, VoterConfig};
