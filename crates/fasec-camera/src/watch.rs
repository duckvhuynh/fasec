//! Camera hot-plug events (WinRT `DeviceWatcher` for `DeviceClass::VideoCapture`, FR-CAM-004).

use crossbeam_channel::Sender;

use crate::CameraError;

/// Ids are normalised with `device::normalize_id`, so they equal `CameraInfo::id`.
/// `Added` is also sent once for every camera present when the watcher starts, and the watcher
/// may report devices that `list()` does not open (e.g. an IR camera): re-`list()` on events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraEvent {
    Added(String),
    Removed(String),
}

/// Running watcher; dropping it unregisters the handlers and stops the watcher.
pub struct WatchGuard {
    _watcher: imp::Watcher,
}

/// Starts watching video capture devices; events are sent to `tx` from a system thread pool.
pub fn watch(tx: Sender<CameraEvent>) -> Result<WatchGuard, CameraError> {
    Ok(WatchGuard {
        _watcher: imp::start(tx)?,
    })
}

#[cfg(windows)]
mod imp {
    use crossbeam_channel::Sender;
    use windows::Devices::Enumeration::{
        DeviceClass, DeviceInformation, DeviceInformationUpdate, DeviceWatcher, DeviceWatcherStatus,
    };
    use windows::Foundation::TypedEventHandler;
    use windows::core::{HSTRING, Ref};

    use super::CameraEvent;
    use crate::{CameraError, normalize_id};

    pub(super) struct Watcher {
        watcher: DeviceWatcher,
        added: i64,
        updated: i64,
        removed: i64,
    }

    fn event(
        id: windows::core::Result<HSTRING>,
        ev: fn(String) -> CameraEvent,
    ) -> Option<CameraEvent> {
        id.ok().map(|id| ev(normalize_id(&id.to_string())))
    }

    pub(super) fn start(tx: Sender<CameraEvent>) -> Result<Watcher, CameraError> {
        let err = |e: windows::core::Error| CameraError::Backend(format!("device watcher: {e}"));
        let watcher =
            DeviceInformation::CreateWatcherDeviceClass(DeviceClass::VideoCapture).map_err(err)?;
        let tx_added = tx.clone();
        let added = watcher
            .Added(&TypedEventHandler::<DeviceWatcher, DeviceInformation>::new(
                move |_, info: Ref<DeviceInformation>| {
                    if let Some(ev) = event(info.ok()?.Id(), CameraEvent::Added) {
                        let _ = tx_added.send(ev); // receiver gone: nobody to tell
                    }
                    Ok(())
                },
            ))
            .map_err(err)?;
        // Without an Updated handler the watcher stops reporting changes after enumeration.
        let updated = watcher
            .Updated(
                &TypedEventHandler::<DeviceWatcher, DeviceInformationUpdate>::new(|_, _| Ok(())),
            )
            .map_err(err)?;
        let removed = watcher
            .Removed(
                &TypedEventHandler::<DeviceWatcher, DeviceInformationUpdate>::new(
                    move |_, info: Ref<DeviceInformationUpdate>| {
                        if let Some(ev) = event(info.ok()?.Id(), CameraEvent::Removed) {
                            let _ = tx.send(ev);
                        }
                        Ok(())
                    },
                ),
            )
            .map_err(err)?;
        let w = Watcher {
            watcher,
            added,
            updated,
            removed,
        };
        w.watcher.Start().map_err(err)?;
        Ok(w)
    }

    impl Drop for Watcher {
        fn drop(&mut self) {
            let w = &self.watcher;
            // Removing the handlers releases their `Sender`s, so receivers see the disconnect.
            let _ = w.RemoveAdded(self.added);
            let _ = w.RemoveUpdated(self.updated);
            let _ = w.RemoveRemoved(self.removed);
            // Stop is only legal while started; any other state is already stopping/stopped.
            let started = matches!(
                w.Status(),
                Ok(DeviceWatcherStatus::Started | DeviceWatcherStatus::EnumerationCompleted)
            );
            if started {
                if let Err(e) = w.Stop() {
                    tracing::warn!(error = %e, "stopping camera device watcher failed");
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use crossbeam_channel::Sender;

    use super::CameraEvent;
    use crate::CameraError;

    pub(super) struct Watcher;

    pub(super) fn start(_: Sender<CameraEvent>) -> Result<Watcher, CameraError> {
        Err(CameraError::Backend(
            "camera hot-plug watching needs Windows".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    #[ignore = "needs Windows device stack"]
    fn watch_starts_and_stops() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let guard = watch(tx).expect("start watcher");
        // The initial enumeration reports every present camera as Added.
        let mut added = Vec::new();
        while let Ok(ev) = rx.recv_timeout(Duration::from_secs(2)) {
            println!("{ev:?}");
            if let CameraEvent::Added(id) = ev {
                added.push(id);
            }
        }
        let listed = crate::list().expect("list cameras");
        for c in &listed {
            assert!(added.contains(&c.id), "{} missing from {added:?}", c.id);
        }
        drop(guard);
        // Stopped: the sender held by the handlers is released.
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(2)),
            Err(crossbeam_channel::RecvTimeoutError::Disconnected)
        ));
    }

    #[test]
    fn watch_guard_can_move_between_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<WatchGuard>();
        assert_send::<CameraEvent>();
    }

    #[cfg(not(windows))]
    #[test]
    fn watch_fails_cleanly_without_windows() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        assert!(matches!(watch(tx), Err(CameraError::Backend(_))));
    }
}
