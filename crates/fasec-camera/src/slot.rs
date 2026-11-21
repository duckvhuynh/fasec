//! Single-slot "latest frame" buffer between the capture thread and the vision worker.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use fasec_core::Frame;

/// Holds at most one frame; a newer `put` replaces an unconsumed older frame.
#[derive(Debug, Default)]
pub struct LatestFrame {
    slot: Mutex<Option<Frame>>,
    cv: Condvar,
}

impl LatestFrame {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `f`, dropping any unconsumed frame. Only holds the mutex for a swap.
    pub fn put(&self, f: Frame) {
        *self.lock() = Some(f);
        self.cv.notify_all();
    }

    /// Waits up to `timeout` for a frame with `seq > after_seq` and takes it out of the slot.
    pub fn take_newer(&self, after_seq: u64, timeout: Duration) -> Option<Frame> {
        let is_newer = |s: &Option<Frame>| s.as_ref().is_some_and(|f| f.seq > after_seq);
        let (mut guard, _) = self
            .cv
            .wait_timeout_while(self.lock(), timeout, |s| !is_newer(s))
            .unwrap_or_else(PoisonError::into_inner);
        if is_newer(&guard) { guard.take() } else { None }
    }

    // A panicking holder cannot leave the slot half-written (it is a plain swap), so poison is ignored.
    fn lock(&self) -> MutexGuard<'_, Option<Frame>> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    fn frame(seq: u64) -> Frame {
        Frame::new(seq, seq, 1, 1, vec![0, 0, 0]).expect("1x1 RGB frame is valid")
    }

    #[test]
    fn slot_take_newer_returns_only_newer() {
        let slot = LatestFrame::new();
        slot.put(frame(5));
        assert!(slot.take_newer(5, Duration::from_millis(10)).is_none());
        assert_eq!(slot.take_newer(4, Duration::ZERO).map(|f| f.seq), Some(5));
        // Taken: never handed out twice.
        assert!(slot.take_newer(0, Duration::from_millis(10)).is_none());
    }

    #[test]
    fn slot_put_replaces_old() {
        let slot = LatestFrame::new();
        for seq in 1..=3 {
            slot.put(frame(seq));
        }
        assert_eq!(slot.take_newer(0, Duration::ZERO).map(|f| f.seq), Some(3));
    }

    #[test]
    fn slot_timeout_returns_none() {
        let slot = LatestFrame::new();
        let t0 = Instant::now();
        assert!(slot.take_newer(0, Duration::from_millis(50)).is_none());
        assert!(t0.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn slot_wakes_waiting_consumer() {
        let slot = Arc::new(LatestFrame::new());
        let producer = {
            let slot = Arc::clone(&slot);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                slot.put(frame(1));
            })
        };
        assert_eq!(
            slot.take_newer(0, Duration::from_secs(2)).map(|f| f.seq),
            Some(1)
        );
        producer.join().expect("producer thread");
    }

    #[test]
    fn slot_producer_never_blocks_on_slow_consumer() {
        let slot = Arc::new(LatestFrame::new());
        let consumer = {
            let slot = Arc::clone(&slot);
            std::thread::spawn(move || {
                let mut last = 0;
                for _ in 0..5 {
                    if let Some(f) = slot.take_newer(last, Duration::from_millis(10)) {
                        assert!(f.seq > last);
                        last = f.seq;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
        };
        let t0 = Instant::now();
        for seq in 1..=1000 {
            slot.put(frame(seq));
        }
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "producer took {elapsed:?}"
        );
        consumer.join().expect("consumer thread");
    }
}
