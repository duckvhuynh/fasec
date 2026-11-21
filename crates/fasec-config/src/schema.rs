//! Config types. `Default` values are the SRS §6.2 example values.

use serde::{Deserialize, Serialize};

/// Highest config `version` this build understands.
pub const CURRENT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub camera: CameraConfig,
    pub sensitivity: SensitivityConfig,
    pub runtime: RuntimeConfig,
    pub owner_absent: OwnerAbsentConfig,
    pub snapshots: SnapshotsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: CURRENT_VERSION,
            camera: CameraConfig::default(),
            sensitivity: SensitivityConfig::default(),
            runtime: RuntimeConfig::default(),
            owner_absent: OwnerAbsentConfig::default(),
            snapshots: SnapshotsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    /// `None` = first available camera.
    pub device_id: Option<String>,
    pub loss_policy: LossPolicy,
    pub resolution: [u32; 2],
    pub fps: u32,
    /// Degrees clockwise; one of 0, 90, 180, 270.
    pub rotation: u16,
    pub mirror: bool,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device_id: None,
            loss_policy: LossPolicy::Protect,
            resolution: [1280, 720],
            fps: 30,
            rotation: 0,
            mirror: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossPolicy {
    #[default]
    Protect,
    FallbackToNext,
    NotifyOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SensitivityConfig {
    pub preset: Preset,
    pub max_distance_m: f32,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            preset: Preset::Balanced,
            max_distance_m: 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    Relaxed,
    #[default]
    Balanced,
    Paranoid,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub execution_provider: ExecutionProvider,
    pub idle_fps: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::Auto,
            idle_fps: 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProvider {
    #[default]
    Auto,
    /// SRS spelling is `directml`, not snake_case `direct_ml`.
    #[serde(rename = "directml")]
    DirectMl,
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OwnerAbsentConfig {
    pub action: OwnerAbsentAction,
    pub delay_s: u32,
}

impl Default for OwnerAbsentConfig {
    fn default() -> Self {
        Self {
            action: OwnerAbsentAction::Off,
            delay_s: 30,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerAbsentAction {
    #[default]
    Off,
    Blur,
    Lock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SnapshotsConfig {
    pub enabled: bool,
    pub retention_days: u32,
    pub min_interval_s: u32,
}

impl Default for SnapshotsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_days: 30,
            min_interval_s: 30,
        }
    }
}
