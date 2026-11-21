//! Config schema, load/save, validation and version migration.

mod schema;

pub use schema::*;

use serde::Deserialize;
use std::io::Write;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("config parse error: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("unsupported config version {0} (this build supports 1..={CURRENT_VERSION})")]
    UnsupportedVersion(i64),
    #[error("invalid config: {0}")]
    Invalid(String),
}

impl Config {
    /// Parses and validates a config document.
    pub fn from_toml(s: &str) -> Result<Config, ConfigError> {
        // Version is checked before the strict parse so a newer file with new
        // fields reports "unsupported version", not "unknown field".
        #[derive(Deserialize)]
        struct Header {
            version: Option<i64>,
        }
        if let Some(v) = toml::from_str::<Header>(s)?.version {
            if v < 1 || v > i64::from(CURRENT_VERSION) {
                return Err(ConfigError::UnsupportedVersion(v));
            }
        }
        // Only version 1 exists, so there is nothing to migrate yet.
        let cfg: Config = toml::from_str(s)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> String {
        // Invariant: every field is a plain TOML-representable value.
        toml::to_string(self).expect("Config always serialises to TOML")
    }

    /// Missing file gives the default config; an unreadable or invalid file is an error.
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(s) => Config::from_toml(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Validates, then writes `<path>.tmp` beside the target and renames it over `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        self.validate()?;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(self.to_toml().as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let invalid = |msg: String| Err(ConfigError::Invalid(msg));
        if self.version < 1 || self.version > CURRENT_VERSION {
            return Err(ConfigError::UnsupportedVersion(i64::from(self.version)));
        }
        let c = &self.camera;
        if ![0, 90, 180, 270].contains(&c.rotation) {
            return invalid(format!(
                "camera.rotation = {} (expected 0, 90, 180 or 270)",
                c.rotation
            ));
        }
        if !(1..=120).contains(&c.fps) {
            return invalid(format!("camera.fps = {} (expected 1..=120)", c.fps));
        }
        let idle = self.runtime.idle_fps;
        if !(1..=30).contains(&idle) {
            return invalid(format!("runtime.idle_fps = {idle} (expected 1..=30)"));
        }
        let d = self.sensitivity.max_distance_m;
        // Written so NaN fails too.
        if !(d > 0.0 && d <= 10.0) {
            return invalid(format!(
                "sensitivity.max_distance_m = {d} (expected (0, 10])"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const SRS_EXAMPLE: &str = r#"
version = 1

[camera]
device_id = "\\\\?\\usb#vid_046d&pid_085e#..."
loss_policy = "protect"            # protect | fallback_to_next | notify_only
resolution = [1280, 720]
fps = 30
rotation = 0
mirror = false

[sensitivity]
preset = "balanced"                # relaxed | balanced | paranoid | custom
max_distance_m = 2.0

[owner_absent]
action = "off"                     # off | blur | lock
delay_s = 30

[snapshots]
enabled = true
retention_days = 30
min_interval_s = 30

[runtime]
execution_provider = "auto"        # auto | directml | cpu
idle_fps = 5
"#;

    /// Fresh per-test directory under the system temp dir.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fasec-config-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn default_roundtrip() {
        let cfg = Config::default();
        assert_eq!(Config::from_toml(&cfg.to_toml()).unwrap(), cfg);
    }

    #[test]
    fn srs_example_parses() {
        let cfg = Config::from_toml(SRS_EXAMPLE).unwrap();
        let expected = Config {
            camera: CameraConfig {
                device_id: Some(r"\\?\usb#vid_046d&pid_085e#...".to_owned()),
                ..CameraConfig::default()
            },
            ..Config::default()
        };
        assert_eq!(cfg, expected);
        assert_eq!(cfg.version, CURRENT_VERSION);
        assert_eq!(cfg.camera.loss_policy, LossPolicy::Protect);
        assert_eq!(cfg.camera.resolution, [1280, 720]);
        assert_eq!(cfg.sensitivity.preset, Preset::Balanced);
        assert_eq!(cfg.runtime.execution_provider, ExecutionProvider::Auto);
        assert_eq!(cfg.owner_absent.action, OwnerAbsentAction::Off);
    }

    #[test]
    fn enums_use_srs_spelling() {
        let cfg = Config::from_toml(
            "[camera]\nloss_policy = \"fallback_to_next\"\n\
             [runtime]\nexecution_provider = \"directml\"\n",
        )
        .unwrap();
        assert_eq!(cfg.camera.loss_policy, LossPolicy::FallbackToNext);
        assert_eq!(cfg.runtime.execution_provider, ExecutionProvider::DirectMl);
        let out = cfg.to_toml();
        assert!(out.contains("\"fallback_to_next\""), "{out}");
        assert!(out.contains("\"directml\""), "{out}");
    }

    #[test]
    fn missing_sections_use_defaults() {
        let cfg = Config::from_toml("version = 1\n[camera]\nfps = 15\n").unwrap();
        assert_eq!(cfg.camera.fps, 15);
        assert_eq!(cfg.camera.resolution, [1280, 720]);
        assert_eq!(cfg.sensitivity, SensitivityConfig::default());
        assert_eq!(cfg.runtime, RuntimeConfig::default());
        assert_eq!(cfg.owner_absent, OwnerAbsentConfig::default());
        assert_eq!(cfg.snapshots, SnapshotsConfig::default());
        assert_eq!(Config::from_toml("").unwrap(), Config::default());
    }

    #[test]
    fn rejects_future_version() {
        let err = Config::from_toml("version = 2\n").unwrap_err();
        assert!(matches!(err, ConfigError::UnsupportedVersion(2)), "{err}");
        let err = Config::from_toml("version = 0\n").unwrap_err();
        assert!(matches!(err, ConfigError::UnsupportedVersion(0)), "{err}");
        // A newer file with fields we don't know yet still reports the version.
        let err = Config::from_toml("version = 2\n[camera]\nhdr = true\n").unwrap_err();
        assert!(matches!(err, ConfigError::UnsupportedVersion(2)), "{err}");
    }

    #[test]
    fn rejects_rotation_45() {
        let err = Config::from_toml("[camera]\nrotation = 45\n").unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)), "{err}");
    }

    #[test]
    fn rejects_out_of_range_values() {
        for toml in [
            "[camera]\nfps = 0\n",
            "[camera]\nfps = 121\n",
            "[runtime]\nidle_fps = 0\n",
            "[runtime]\nidle_fps = 31\n",
            "[sensitivity]\nmax_distance_m = 0.0\n",
            "[sensitivity]\nmax_distance_m = 10.5\n",
            "[sensitivity]\nmax_distance_m = nan\n",
        ] {
            let err = Config::from_toml(toml).unwrap_err();
            assert!(matches!(err, ConfigError::Invalid(_)), "{toml}: {err}");
        }
        for toml in [
            "[camera]\nfps = 120\nrotation = 270\n",
            "[runtime]\nidle_fps = 30\n",
            "[sensitivity]\nmax_distance_m = 10.0\n",
        ] {
            Config::from_toml(toml).unwrap();
        }
    }

    #[test]
    fn rejects_unknown_field() {
        let err = Config::from_toml("[camera]\nfpz = 30\n").unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)), "{err}");
        let err = Config::from_toml("bogus = 1\n").unwrap_err();
        assert!(matches!(err, ConfigError::Parse(_)), "{err}");
    }

    #[test]
    fn load_missing_file_gives_default() {
        let path = temp_dir("missing").join("config.toml");
        assert_eq!(Config::load_or_default(&path).unwrap(), Config::default());
    }

    #[test]
    fn load_invalid_file_is_error() {
        let path = temp_dir("invalid").join("config.toml");
        std::fs::write(&path, "[camera]\nrotation = 45\n").unwrap();
        assert!(Config::load_or_default(&path).is_err());
        std::fs::write(&path, "not toml [[[").unwrap();
        assert!(matches!(
            Config::load_or_default(&path),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn save_then_load_equal() {
        let dir = temp_dir("save");
        let path = dir.join("config.toml");
        let mut cfg = Config::default();
        cfg.camera.device_id = Some("cam-1".to_owned());
        cfg.camera.rotation = 90;
        cfg.owner_absent.action = OwnerAbsentAction::Lock;
        cfg.save(&path).unwrap();
        assert_eq!(Config::load_or_default(&path).unwrap(), cfg);
        // Overwrite an existing file; no temp file left behind.
        cfg.snapshots.enabled = false;
        cfg.save(&path).unwrap();
        assert_eq!(Config::load_or_default(&path).unwrap(), cfg);
        assert!(!dir.join("config.toml.tmp").exists());
    }
}
