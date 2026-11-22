//! Enrolment flows, face gallery and encrypted template store.

pub mod error;
mod pin;
pub mod protect;
pub mod store;

pub use error::IdentityError;
pub use protect::{ProtectError, SecretProtector};
pub use store::{
    DecryptedTemplate, EMBEDDING_DIM, Gallery, Person, PersonId, Pose, Role, StoredTemplate,
};
