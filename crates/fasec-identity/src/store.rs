//! Encrypted SQLite gallery (`gallery.db`, SRS §8.1).

use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, ErrorCode, params};
use zeroize::Zeroizing;

use crate::error::IdentityError;
use crate::protect::{ProtectError, SecretProtector};

type Result<T> = std::result::Result<T, IdentityError>;

/// Template dimensionality (SFace, FR-ID-003).
pub const EMBEDDING_DIM: usize = 128;

const SCHEMA: &str = include_str!("schema.sql");
const SCHEMA_VERSION: u32 = 1;

/// One decrypted template: owner, role and embedding (zeroised on drop).
pub type DecryptedTemplate = (PersonId, Role, Zeroizing<Vec<f32>>);

/// UUID v4 string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PersonId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Owner,
    Authorised,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Owner => "owner",
            Role::Authorised => "authorised",
        }
    }

    fn parse(s: &str) -> Result<Self> {
        match s {
            "owner" => Ok(Role::Owner),
            "authorised" => Ok(Role::Authorised),
            _ => Err(IdentityError::Db(format!("unknown role {s:?}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pose {
    Front,
    Left,
    Right,
    Up,
    Down,
}

impl Pose {
    fn as_str(self) -> &'static str {
        match self {
            Pose::Front => "front",
            Pose::Left => "left",
            Pose::Right => "right",
            Pose::Up => "up",
            Pose::Down => "down",
        }
    }

    fn parse(s: &str) -> Result<Self> {
        match s {
            "front" => Ok(Pose::Front),
            "left" => Ok(Pose::Left),
            "right" => Ok(Pose::Right),
            "up" => Ok(Pose::Up),
            "down" => Ok(Pose::Down),
            _ => Err(IdentityError::Db(format!("unknown pose {s:?}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Person {
    pub id: PersonId,
    pub name: String,
    pub role: Role,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Template metadata only; the embedding is never part of this type.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredTemplate {
    pub id: i64,
    pub person: PersonId,
    pub model_id: String,
    pub pose: Pose,
    pub quality: f32,
    pub created_at: i64,
}

/// The face gallery. Template blobs go through `protector` on every write/read.
pub struct Gallery {
    conn: Connection,
    protector: Arc<dyn SecretProtector>,
}

impl Gallery {
    pub fn open(path: &Path, protector: Arc<dyn SecretProtector>) -> Result<Gallery> {
        Self::init(Connection::open(path)?, protector)
    }

    pub fn open_in_memory(protector: Arc<dyn SecretProtector>) -> Result<Gallery> {
        Self::init(Connection::open_in_memory()?, protector)
    }

    fn init(mut conn: Connection, protector: Arc<dyn SecretProtector>) -> Result<Gallery> {
        conn.pragma_update(None, "foreign_keys", true)?;
        // Overwrite freed pages so deleted ciphertext does not linger in the file.
        conn.pragma_update(None, "secure_delete", true)?;
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        match u32::try_from(version).unwrap_or(u32::MAX) {
            0 => {
                let tx = conn.transaction()?;
                tx.execute_batch(SCHEMA)?;
                tx.commit()?;
                tracing::info!(version = SCHEMA_VERSION, "gallery schema created");
            }
            SCHEMA_VERSION => {}
            v => return Err(IdentityError::UnsupportedSchema(v)),
        }
        Ok(Gallery { conn, protector })
    }

    pub fn add_person(&mut self, name: &str, role: Role) -> Result<PersonId> {
        let id = PersonId(uuid::Uuid::new_v4().to_string());
        let now = now();
        let res = self.conn.execute(
            "INSERT INTO person (id, name, role, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?4)",
            params![id.0, name, role.as_str(), now],
        );
        match res {
            Err(e) if role == Role::Owner && is_constraint(&e) => Err(IdentityError::OwnerExists),
            Err(e) => Err(e.into()),
            Ok(_) => {
                tracing::info!(person = %id.0, role = role.as_str(), "person added");
                Ok(id)
            }
        }
    }

    pub fn persons(&self) -> Result<Vec<Person>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, role, created_at, updated_at FROM person ORDER BY created_at, id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?;
        rows.map(|row| {
            let (id, name, role, created_at, updated_at) = row?;
            Ok(Person {
                id: PersonId(id),
                name,
                role: Role::parse(&role)?,
                created_at,
                updated_at,
            })
        })
        .collect()
    }

    /// Removes a person and (by cascade) all their templates.
    pub fn remove_person(&mut self, id: &PersonId) -> Result<()> {
        match self
            .conn
            .execute("DELETE FROM person WHERE id = ?1", [&id.0])?
        {
            0 => Err(IdentityError::NotFound),
            _ => {
                tracing::info!(person = %id.0, "person removed");
                Ok(())
            }
        }
    }

    /// Encrypts `embedding` with the protector and stores it. Returns the template id.
    pub fn add_template(
        &mut self,
        person: &PersonId,
        model_id: &str,
        pose: Pose,
        quality: f32,
        embedding: &[f32],
    ) -> Result<i64> {
        if embedding.len() != EMBEDDING_DIM || !embedding.iter().all(|v| v.is_finite()) {
            return Err(IdentityError::InvalidEmbedding);
        }
        let mut plain = Zeroizing::new(Vec::with_capacity(EMBEDDING_DIM * 4));
        for v in embedding {
            plain.extend_from_slice(&v.to_le_bytes());
        }
        let blob = self.protector.protect(&plain)?;
        let res = self.conn.execute(
            "INSERT INTO template (person_id, model_id, blob, pose, quality, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                person.0,
                model_id,
                blob,
                pose.as_str(),
                f64::from(quality),
                now()
            ],
        );
        match res {
            // Only the FK on person_id can be violated here.
            Err(e) if is_constraint(&e) => Err(IdentityError::NotFound),
            Err(e) => Err(e.into()),
            Ok(_) => Ok(self.conn.last_insert_rowid()),
        }
    }

    /// Metadata of one person's templates (no embeddings).
    pub fn templates(&self, person: &PersonId) -> Result<Vec<StoredTemplate>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model_id, pose, quality, created_at FROM template
             WHERE person_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map([&person.0], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, f64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?;
        rows.map(|row| {
            let (id, model_id, pose, quality, created_at) = row?;
            Ok(StoredTemplate {
                id,
                person: person.clone(),
                model_id,
                pose: Pose::parse(&pose)?,
                quality: quality as f32,
                created_at,
            })
        })
        .collect()
    }

    /// Decrypts every template enrolled with `model_id`. Any undecryptable or
    /// wrong-length row fails the whole call (fail safe: never match on a partial gallery).
    pub fn templates_for_model(&self, model_id: &str) -> Result<Vec<DecryptedTemplate>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.id, t.person_id, p.role, t.blob FROM template t
             JOIN person p ON p.id = t.person_id WHERE t.model_id = ?1 ORDER BY t.id",
        )?;
        let mut rows = stmt.query([model_id])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let template_id: i64 = r.get(0)?;
            let person = PersonId(r.get(1)?);
            let role = Role::parse(&r.get::<_, String>(2)?)?;
            let blob: Vec<u8> = r.get(3)?;
            let corrupt = || {
                tracing::warn!(template_id, "gallery template failed to load");
                IdentityError::CorruptTemplate { template_id }
            };
            let plain = match self.protector.unprotect(&blob) {
                Ok(p) => p,
                Err(ProtectError::Unsupported) => return Err(ProtectError::Unsupported.into()),
                Err(_) => return Err(corrupt()),
            };
            if plain.len() != EMBEDDING_DIM * 4 {
                return Err(corrupt());
            }
            let mut emb = Zeroizing::new(Vec::with_capacity(EMBEDDING_DIM));
            emb.extend(
                plain
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c)),
            );
            out.push((person, role, emb));
        }
        Ok(out)
    }

    pub fn model_ids_in_use(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT model_id FROM template ORDER BY model_id")?;
        let ids = stmt.query_map([], |r| r.get(0))?;
        Ok(ids.collect::<rusqlite::Result<_>>()?)
    }

    /// FR-ID-007: erases persons, templates, calibration and the PIN, then compacts the file.
    pub fn delete_all_biometrics(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            "DELETE FROM template; DELETE FROM person; DELETE FROM calibration;
             DELETE FROM secret WHERE kind = 'pin';",
        )?;
        tx.commit()?;
        self.conn.execute_batch("VACUUM")?;
        tracing::info!("all biometric data deleted");
        Ok(())
    }
}

fn is_constraint(e: &rusqlite::Error) -> bool {
    e.sqlite_error_code() == Some(ErrorCode::ConstraintViolation)
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protect::testing::XorProtector;

    const MODEL: &str = "sface-2021dec";

    fn gallery() -> Gallery {
        Gallery::open_in_memory(Arc::new(XorProtector::default())).unwrap()
    }

    fn embedding(seed: f32) -> Vec<f32> {
        (0..EMBEDDING_DIM).map(|i| seed + i as f32 * 0.01).collect()
    }

    fn count(g: &Gallery, table: &str) -> i64 {
        g.conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn schema_applies_and_sets_version() {
        let g = gallery();
        let v: i64 = g
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(v, 1);
        let fk: i64 = g
            .conn
            .pragma_query_value(None, "foreign_keys", |r| r.get(0))
            .unwrap();
        assert_eq!(fk, 1);
        for t in ["person", "template", "calibration", "secret"] {
            assert_eq!(count(&g, t), 0, "{t}");
        }
    }

    #[test]
    fn rejects_future_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gallery.db");
        let conn = Connection::open(&path).unwrap();
        conn.pragma_update(None, "user_version", 2).unwrap();
        drop(conn);
        let err = Gallery::open(&path, Arc::new(XorProtector::default())).err();
        assert!(
            matches!(err, Some(IdentityError::UnsupportedSchema(2))),
            "{err:?}"
        );
    }

    #[test]
    fn only_one_owner() {
        let mut g = gallery();
        g.add_person("Alice", Role::Owner).unwrap();
        assert!(matches!(
            g.add_person("Mallory", Role::Owner),
            Err(IdentityError::OwnerExists)
        ));
        g.add_person("Bob", Role::Authorised).unwrap();
        g.add_person("Carol", Role::Authorised).unwrap();
        let persons = g.persons().unwrap();
        assert_eq!(persons.len(), 3);
        assert_eq!(persons.iter().filter(|p| p.role == Role::Owner).count(), 1);
    }

    #[test]
    fn template_roundtrip() {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        let bob = g.add_person("Bob", Role::Authorised).unwrap();
        let (ea, eb) = (embedding(0.5), embedding(-0.25));
        g.add_template(&alice, MODEL, Pose::Front, 0.9, &ea)
            .unwrap();
        g.add_template(&bob, MODEL, Pose::Left, 0.8, &eb).unwrap();

        let mut got = g.templates_for_model(MODEL).unwrap();
        got.sort_by_key(|(_, role, _)| *role != Role::Owner);
        assert_eq!(got.len(), 2);
        assert_eq!((&got[0].0, got[0].1), (&alice, Role::Owner));
        assert_eq!(&got[0].2[..], &ea[..]);
        assert_eq!((&got[1].0, got[1].1), (&bob, Role::Authorised));
        assert_eq!(&got[1].2[..], &eb[..]);

        let meta = g.templates(&bob).unwrap();
        assert_eq!(meta.len(), 1);
        assert_eq!(
            (meta[0].pose, meta[0].model_id.as_str()),
            (Pose::Left, MODEL)
        );
        assert!((meta[0].quality - 0.8).abs() < 1e-6);
    }

    #[test]
    fn add_template_rejects_bad_embedding_and_unknown_person() {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        let short = vec![0.1; EMBEDDING_DIM - 1];
        let mut nan = embedding(0.0);
        nan[3] = f32::NAN;
        for e in [&short, &nan] {
            assert!(matches!(
                g.add_template(&alice, MODEL, Pose::Front, 0.9, e),
                Err(IdentityError::InvalidEmbedding)
            ));
        }
        let ghost = PersonId("no-such-person".into());
        assert!(matches!(
            g.add_template(&ghost, MODEL, Pose::Front, 0.9, &embedding(0.0)),
            Err(IdentityError::NotFound)
        ));
        assert_eq!(count(&g, "template"), 0);
    }

    #[test]
    fn template_blob_is_not_plaintext() {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        let e = embedding(0.5);
        g.add_template(&alice, MODEL, Pose::Front, 0.9, &e).unwrap();
        let blob: Vec<u8> = g
            .conn
            .query_row("SELECT blob FROM template", [], |r| r.get(0))
            .unwrap();
        let plain: Vec<u8> = e.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert!(!blob.is_empty());
        assert!(!blob.windows(plain.len()).any(|w| w == plain.as_slice()));
        // No single 4-byte value of the embedding leaks verbatim either.
        assert!(!blob.windows(4).any(|w| w == &plain[..4]));
    }

    #[test]
    fn remove_person_cascades_templates() {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        let bob = g.add_person("Bob", Role::Authorised).unwrap();
        g.add_template(&alice, MODEL, Pose::Front, 0.9, &embedding(0.1))
            .unwrap();
        g.add_template(&alice, MODEL, Pose::Up, 0.9, &embedding(0.2))
            .unwrap();
        g.add_template(&bob, MODEL, Pose::Down, 0.9, &embedding(0.3))
            .unwrap();

        g.remove_person(&alice).unwrap();
        assert_eq!(count(&g, "template"), 1);
        let left = g.templates_for_model(MODEL).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].0, bob);
        assert!(matches!(
            g.remove_person(&alice),
            Err(IdentityError::NotFound)
        ));
        // The owner slot is free again.
        g.add_person("Alice again", Role::Owner).unwrap();
    }

    #[test]
    fn templates_for_other_model_not_returned() {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        g.add_template(&alice, "old-model", Pose::Front, 0.9, &embedding(0.1))
            .unwrap();
        g.add_template(&alice, MODEL, Pose::Front, 0.9, &embedding(0.2))
            .unwrap();

        let got = g.templates_for_model(MODEL).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(&got[0].2[..], &embedding(0.2)[..]);
        assert!(g.templates_for_model("never-used").unwrap().is_empty());
        assert_eq!(g.model_ids_in_use().unwrap(), vec!["old-model", MODEL]);
    }

    #[test]
    fn delete_all_biometrics_empties_everything() {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        g.add_template(&alice, MODEL, Pose::Front, 0.9, &embedding(0.1))
            .unwrap();
        g.conn
            .execute(
                "INSERT INTO calibration VALUES ('cam', -30, 30, -20, 20, 0)",
                [],
            )
            .unwrap();
        g.conn
            .execute("INSERT INTO secret VALUES ('pin', '$argon2id$x')", [])
            .unwrap();

        g.delete_all_biometrics().unwrap();
        for t in ["person", "template", "calibration", "secret"] {
            assert_eq!(count(&g, t), 0, "{t}");
        }
        assert!(g.model_ids_in_use().unwrap().is_empty());
        // Gallery stays usable afterwards.
        g.add_person("Alice", Role::Owner).unwrap();
    }

    #[test]
    fn file_db_persists_across_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gallery.db");
        let protector = Arc::new(XorProtector::default());
        let (alice, e) = {
            let mut g = Gallery::open(&path, protector.clone()).unwrap();
            let alice = g.add_person("Alice", Role::Owner).unwrap();
            let e = embedding(0.7);
            g.add_template(&alice, MODEL, Pose::Right, 0.95, &e)
                .unwrap();
            (alice, e)
        };
        let g = Gallery::open(&path, protector).unwrap();
        let persons = g.persons().unwrap();
        assert_eq!(persons.len(), 1);
        assert_eq!(
            (&persons[0].id, persons[0].name.as_str()),
            (&alice, "Alice")
        );
        assert!(persons[0].created_at > 0);
        let got = g.templates_for_model(MODEL).unwrap();
        assert_eq!(&got[0].2[..], &e[..]);
    }
}
