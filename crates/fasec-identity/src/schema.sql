-- gallery.db schema v1 (SRS §8.1). Applied once when user_version = 0.
CREATE TABLE person (
    id         TEXT PRIMARY KEY NOT NULL,
    name       TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('owner', 'authorised')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
) STRICT;

-- Exactly one owner.
CREATE UNIQUE INDEX one_owner ON person(role) WHERE role = 'owner';

CREATE TABLE template (
    id         INTEGER PRIMARY KEY,
    person_id  TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    model_id   TEXT NOT NULL,
    blob       BLOB NOT NULL, -- SecretProtector ciphertext of little-endian f32s; never plaintext
    pose       TEXT NOT NULL CHECK (pose IN ('front', 'left', 'right', 'up', 'down')),
    quality    REAL NOT NULL,
    created_at INTEGER NOT NULL
) STRICT;

CREATE INDEX template_person ON template(person_id);
CREATE INDEX template_model ON template(model_id);

CREATE TABLE calibration (
    camera_id  TEXT PRIMARY KEY NOT NULL,
    yaw_min    REAL NOT NULL,
    yaw_max    REAL NOT NULL,
    pitch_min  REAL NOT NULL,
    pitch_max  REAL NOT NULL,
    created_at INTEGER NOT NULL
) STRICT;

CREATE TABLE secret (
    kind TEXT PRIMARY KEY NOT NULL, -- 'pin'
    hash TEXT NOT NULL              -- Argon2id PHC string
) STRICT;

PRAGMA user_version = 1;
