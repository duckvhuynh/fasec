//! Camera health from frame arrival times and brightness (platform-neutral, clock passed in).

use fasec_core::CameraHealth;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthConfig {
    /// No frame for longer than this means `Lost` (FR-CAM-005).
    pub frame_timeout_ms: u64,
    /// Mean luma below this counts as black.
    pub black_luma: f32,
    /// Black for longer than this means `Covered` (FR-CAM-009).
    pub black_ms: u64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        HealthConfig {
            frame_timeout_ms: 1500,
            black_luma: 10.0,
            black_ms: 2000,
        }
    }
}

/// Tracks one capture session. `now_ms` is a monotonic millisecond clock whose 0 is when the
/// capture was started; before the first frame the camera counts as `Lost` once `now_ms`
/// exceeds `frame_timeout_ms` (any other clock origin only makes that happen sooner).
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    cfg: HealthConfig,
    /// Arrival time of the latest frame (0 = capture start until the first frame).
    last_frame_ms: u64,
    /// Arrival time of the first frame of the current black streak.
    black_since_ms: Option<u64>,
}

impl HealthMonitor {
    pub fn new(cfg: HealthConfig) -> Self {
        HealthMonitor {
            cfg,
            last_frame_ms: 0,
            black_since_ms: None,
        }
    }

    pub fn on_frame(&mut self, now_ms: u64, mean_luma: f32) {
        self.last_frame_ms = now_ms;
        if mean_luma < self.cfg.black_luma {
            self.black_since_ms.get_or_insert(now_ms);
        } else {
            self.black_since_ms = None;
        }
    }

    /// `Lost` beats `Covered`: a camera that stopped delivering says nothing about the scene.
    pub fn evaluate(&self, now_ms: u64) -> CameraHealth {
        if now_ms.saturating_sub(self.last_frame_ms) > self.cfg.frame_timeout_ms {
            CameraHealth::Lost
        } else if self
            .black_since_ms
            .is_some_and(|t| now_ms.saturating_sub(t) > self.cfg.black_ms)
        {
            CameraHealth::Covered
        } else {
            CameraHealth::Ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRIGHT: f32 = 120.0;
    const BLACK: f32 = 3.0;

    fn mon() -> HealthMonitor {
        HealthMonitor::new(HealthConfig::default())
    }

    /// Feeds frames every `step` ms from `from` to `to` (inclusive) with `luma`.
    fn feed(m: &mut HealthMonitor, from: u64, to: u64, step: u64, luma: f32) {
        for t in (from..=to).step_by(step as usize) {
            m.on_frame(t, luma);
        }
    }

    #[test]
    fn defaults_match_srs() {
        let c = HealthConfig::default();
        assert_eq!(
            (c.frame_timeout_ms, c.black_luma, c.black_ms),
            (1500, 10.0, 2000)
        );
    }

    #[test]
    fn health_ok_with_regular_frames() {
        let mut m = mon();
        for t in (0..=10_000).step_by(33) {
            m.on_frame(t, BRIGHT);
            assert_eq!(m.evaluate(t + 10), CameraHealth::Ok, "at {t}");
        }
    }

    #[test]
    fn health_lost_after_timeout() {
        let mut m = mon();
        feed(&mut m, 0, 3000, 33, BRIGHT); // last frame at 2970
        assert_eq!(m.evaluate(2970 + 1500), CameraHealth::Ok);
        assert_eq!(m.evaluate(2970 + 1501), CameraHealth::Lost);
        // A late frame brings it back.
        m.on_frame(5000, BRIGHT);
        assert_eq!(m.evaluate(5001), CameraHealth::Ok);
    }

    #[test]
    fn health_lost_if_never_started() {
        let m = mon();
        assert_eq!(
            m.evaluate(1500),
            CameraHealth::Ok,
            "grace period after start"
        );
        assert_eq!(m.evaluate(1501), CameraHealth::Lost);
        assert_eq!(m.evaluate(60_000), CameraHealth::Lost);
    }

    #[test]
    fn health_covered_after_2s_black() {
        let mut m = mon();
        feed(&mut m, 0, 990, 33, BRIGHT);
        feed(&mut m, 1000, 3000, 33, BLACK); // black since 1000, last frame 2980
        assert_eq!(m.evaluate(2990), CameraHealth::Ok, "black for 1990 ms");
        assert_eq!(
            m.evaluate(3000),
            CameraHealth::Ok,
            "black for exactly 2000 ms"
        );
        assert_eq!(m.evaluate(3001), CameraHealth::Covered);
    }

    #[test]
    fn health_lost_wins_over_covered() {
        let mut m = mon();
        feed(&mut m, 0, 3000, 33, BLACK);
        assert_eq!(m.evaluate(3000), CameraHealth::Covered);
        assert_eq!(m.evaluate(2970 + 1501), CameraHealth::Lost);
    }

    #[test]
    fn health_recovers_after_light() {
        let mut m = mon();
        feed(&mut m, 0, 3000, 33, BLACK);
        assert_eq!(m.evaluate(3000), CameraHealth::Covered);
        m.on_frame(3033, BRIGHT);
        assert_eq!(m.evaluate(3033), CameraHealth::Ok);
        // A new black streak starts the 2 s count again.
        feed(&mut m, 3066, 5000, 33, BLACK);
        assert_eq!(m.evaluate(5000), CameraHealth::Ok);
        assert_eq!(m.evaluate(5067), CameraHealth::Covered);
    }

    #[test]
    fn health_black_shorter_than_2s_is_ok() {
        let mut m = mon();
        for cycle in 0..5u64 {
            let t0 = cycle * 3000;
            feed(&mut m, t0, t0 + 1900, 33, BLACK);
            assert_eq!(m.evaluate(t0 + 1990), CameraHealth::Ok);
            feed(&mut m, t0 + 2000, t0 + 2990, 33, BRIGHT);
            assert_eq!(m.evaluate(t0 + 2990), CameraHealth::Ok);
        }
    }

    #[test]
    fn luma_at_threshold_is_not_black() {
        let mut m = mon();
        feed(&mut m, 0, 5000, 33, 10.0);
        assert_eq!(m.evaluate(5000), CameraHealth::Ok);
    }

    #[test]
    fn clock_going_backwards_does_not_panic() {
        let mut m = mon();
        m.on_frame(5000, BRIGHT);
        assert_eq!(m.evaluate(4000), CameraHealth::Ok);
    }
}
