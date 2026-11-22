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
///
/// Security limit: a ciphertext is not bound to its row, person or model id. Under
/// DPAPI current-user scope any process running as the same user can produce valid
/// template blobs (or move them between rows). The gallery protects templates at rest
/// from other users and offline copies of `gallery.db`, not from a same-user attacker.
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

    /// Decrypts every template enrolled with `model_id`. Any undecryptable, wrong-length
    /// or non-finite row fails the whole call (fail safe: never match on a partial gallery).
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
            if !emb.iter().all(|v| v.is_finite()) {
                return Err(corrupt());
            }
            out.push((person, role, emb));
        }
        Ok(out)
    }

    /// Sets or replaces the owner PIN (4–12 ASCII digits), stored as an Argon2id PHC string.
    pub fn set_pin(&mut self, pin: &str) -> Result<()> {
        let phc = Zeroizing::new(crate::pin::hash(pin)?);
        self.conn.execute(
            "INSERT INTO secret (kind, hash) VALUES ('pin', ?1)
             ON CONFLICT (kind) DO UPDATE SET hash = excluded.hash",
            [phc.as_str()],
        )?;
        tracing::info!("owner PIN set");
        Ok(())
    }

    /// `Ok(false)` on a wrong or malformed PIN, or when no PIN is set. Errors only on
    /// storage problems or an unparsable stored hash.
    pub fn verify_pin(&self, pin: &str) -> Result<bool> {
        match self.stored_pin_hash()? {
            Some(phc) => crate::pin::verify(pin, &phc),
            None => Ok(false),
        }
    }

    pub fn has_pin(&self) -> Result<bool> {
        Ok(self.stored_pin_hash()?.is_some())
    }

    fn stored_pin_hash(&self) -> Result<Option<Zeroizing<String>>> {
        let res = self
            .conn
            .query_row("SELECT hash FROM secret WHERE kind = 'pin'", [], |r| {
                r.get::<_, String>(0)
            });
        match res {
            Ok(h) => Ok(Some(Zeroizing::new(h))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
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

    fn stored_pin(g: &Gallery) -> String {
        g.conn
            .query_row("SELECT hash FROM secret WHERE kind = 'pin'", [], |r| {
                r.get(0)
            })
            .unwrap()
    }

    #[test]
    fn pin_roundtrip() {
        let mut g = gallery();
        assert!(!g.has_pin().unwrap());
        assert!(!g.verify_pin("1234").unwrap(), "no PIN set must not verify");
        g.set_pin("1234").unwrap();
        assert!(g.has_pin().unwrap());
        assert!(g.verify_pin("1234").unwrap());
    }

    #[test]
    fn wrong_pin_is_false() {
        let mut g = gallery();
        g.set_pin("246813").unwrap();
        assert!(!g.verify_pin("246814").unwrap());
        // Malformed input is simply not the PIN.
        assert!(!g.verify_pin("").unwrap());
        assert!(!g.verify_pin("abcd").unwrap());
    }

    #[test]
    fn pin_rules_rejected() {
        let mut g = gallery();
        for bad in ["", "123", "1234567890123", "12a4", "١٢٣٤", " 1234", "12 34"] {
            assert!(
                matches!(g.set_pin(bad), Err(IdentityError::Pin(_))),
                "{bad:?} accepted"
            );
        }
        assert!(!g.has_pin().unwrap());
    }

    #[test]
    fn pin_hash_uses_argon2id_params() {
        let mut g = gallery();
        g.set_pin("123456789012").unwrap();
        let phc = stored_pin(&g);
        assert!(phc.starts_with("$argon2id$v=19$"), "not argon2id v19");
        let parsed = argon2::PasswordHash::new(&phc).unwrap();
        let params = argon2::Params::try_from(&parsed).unwrap();
        assert!(params.m_cost() >= 19456);
        assert!(params.t_cost() >= 2);
        assert_eq!(params.p_cost(), 1);
        assert_eq!(parsed.salt.unwrap().as_ref().len(), 16);
        assert!(!phc.contains("123456789012"));
    }

    #[test]
    fn pin_rehash_replaces_old() {
        let mut g = gallery();
        g.set_pin("1111").unwrap();
        let first = stored_pin(&g);
        g.set_pin("2222").unwrap();
        assert_eq!(count(&g, "secret"), 1);
        assert_ne!(stored_pin(&g), first);
        assert!(g.verify_pin("2222").unwrap());
        assert!(!g.verify_pin("1111").unwrap());
    }

    #[test]
    fn corrupt_pin_hash_is_error_not_false_or_true() {
        let g = gallery();
        g.conn
            .execute("INSERT INTO secret VALUES ('pin', 'not a phc string')", [])
            .unwrap();
        assert!(matches!(g.verify_pin("1234"), Err(IdentityError::Pin(_))));
    }

    /// Two good templates around `bad`; returns (gallery, id of the middle row).
    fn gallery_with_three_templates() -> (Gallery, i64) {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        let bob = g.add_person("Bob", Role::Authorised).unwrap();
        g.add_template(&alice, MODEL, Pose::Front, 0.9, &embedding(0.1))
            .unwrap();
        let mid = g
            .add_template(&alice, MODEL, Pose::Left, 0.9, &embedding(0.2))
            .unwrap();
        g.add_template(&bob, MODEL, Pose::Front, 0.9, &embedding(0.3))
            .unwrap();
        assert_eq!(g.templates_for_model(MODEL).unwrap().len(), 3);
        (g, mid)
    }

    fn blob(g: &Gallery, id: i64) -> Vec<u8> {
        g.conn
            .query_row("SELECT blob FROM template WHERE id = ?1", [id], |r| {
                r.get(0)
            })
            .unwrap()
    }

    fn set_blob(g: &Gallery, id: i64, blob: &[u8]) {
        g.conn
            .execute(
                "UPDATE template SET blob = ?1 WHERE id = ?2",
                params![blob, id],
            )
            .unwrap();
    }

    fn assert_corrupt(g: &Gallery, id: i64) {
        let res = g.templates_for_model(MODEL);
        assert!(
            matches!(res, Err(IdentityError::CorruptTemplate { template_id }) if template_id == id),
            "{:?}",
            res.map(|v| v.len())
        );
    }

    #[test]
    fn tampered_template_blob_fails_load() {
        let (g, id) = gallery_with_three_templates();
        let mut b = blob(&g, id);
        b[17] ^= 0x01;
        set_blob(&g, id, &b);
        assert_corrupt(&g, id);
    }

    #[test]
    fn truncated_template_blob_fails_load() {
        let (g, id) = gallery_with_three_templates();
        let b = blob(&g, id);
        set_blob(&g, id, &b[..b.len() - 1]);
        assert_corrupt(&g, id);
        set_blob(&g, id, &[]);
        assert_corrupt(&g, id);
        // Authentic ciphertext of a too-short plaintext is still rejected.
        let short = XorProtector::default().protect(&[0u8; 4 * 127]).unwrap();
        set_blob(&g, id, &short);
        assert_corrupt(&g, id);
    }

    #[test]
    fn blob_from_other_protector_fails_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gallery.db");
        let id = {
            let mut g = Gallery::open(&path, Arc::new(XorProtector(1))).unwrap();
            let alice = g.add_person("Alice", Role::Owner).unwrap();
            g.add_template(&alice, MODEL, Pose::Front, 0.9, &embedding(0.1))
                .unwrap()
        };
        let g = Gallery::open(&path, Arc::new(XorProtector(2))).unwrap();
        assert_corrupt(&g, id);
    }

    #[test]
    fn one_bad_row_fails_whole_load() {
        // One row holds a blob from another key: the two good rows must not be returned alone.
        let (g, id) = gallery_with_three_templates();
        let foreign = XorProtector(0x33)
            .protect(&[0u8; 4 * EMBEDDING_DIM])
            .unwrap();
        set_blob(&g, id, &foreign);
        assert_corrupt(&g, id);
    }

    #[test]
    fn non_finite_template_value_fails_load() {
        let (g, id) = gallery_with_three_templates();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut e = embedding(0.2);
            e[64] = bad;
            let plain: Vec<u8> = e.iter().flat_map(|x| x.to_le_bytes()).collect();
            set_blob(&g, id, &XorProtector::default().protect(&plain).unwrap());
            assert_corrupt(&g, id);
        }
    }

    // ---- audit additions ----

    #[test]
    #[ignore = "BUG: non-finite quality binds as SQL NULL, violates NOT NULL and is reported as NotFound for an existing person (store.rs:222-224)"]
    fn nan_quality_is_not_reported_as_missing_person() {
        let mut g = gallery();
        let alice = g.add_person("Alice", Role::Owner).unwrap();
        let res = g.add_template(&alice, MODEL, Pose::Front, f32::NAN, &embedding(0.1));
        assert!(
            !matches!(res, Err(IdentityError::NotFound)),
            "misleading error: {res:?}"
        );
        assert!(res.is_err() || count(&g, "template") == 1);
    }

    #[test]
    fn delete_all_leaves_no_pin_or_ciphertext_in_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gallery.db");
        let mut g = Gallery::open(&path, Arc::new(XorProtector::default())).unwrap();
        let alice = g.add_person("Alice Distinctname", Role::Owner).unwrap();
        let tid = g
            .add_template(&alice, MODEL, Pose::Front, 0.9, &embedding(0.1))
            .unwrap();
        g.set_pin("1234").unwrap();
        let blob = blob(&g, tid);
        let phc = stored_pin(&g);
        let in_file = |needle: &[u8]| {
            std::fs::read(&path)
                .unwrap()
                .windows(needle.len())
                .any(|w| w == needle)
        };
        assert!(in_file(phc.as_bytes()) && in_file(&blob), "fixture sanity");

        g.delete_all_biometrics().unwrap();
        assert!(!g.has_pin().unwrap());
        assert!(!g.verify_pin("1234").unwrap());
        assert!(g.persons().unwrap().is_empty());
        assert!(!in_file(phc.as_bytes()), "PIN hash survived in file");
        assert!(
            !in_file(&blob[..16]),
            "template ciphertext survived in file"
        );
        assert!(
            !in_file(b"Alice Distinctname"),
            "person name survived in file"
        );
        for ext in ["db-wal", "db-journal"] {
            let side = dir.path().join(format!("gallery.{ext}"));
            assert!(!side.exists() || std::fs::metadata(side).unwrap().len() == 0);
        }
    }
}
