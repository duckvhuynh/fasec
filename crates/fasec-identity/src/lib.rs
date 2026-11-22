//! Enrolment flows, face gallery and encrypted template store.

pub mod error;
pub mod matcher;
pub mod protect;
pub mod store;
pub mod voter;

pub use error::IdentityError;
pub use matcher::{MatchResult, Matcher, MatcherConfig};
pub use protect::{ProtectError, SecretProtector};
pub use store::{
    DecryptedTemplate, EMBEDDING_DIM, Gallery, Person, PersonId, Pose, Role, StoredTemplate,
};
pub use voter::{Voter, VoterConfig};
