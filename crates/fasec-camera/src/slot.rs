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

    #[test]
    fn slot_keeps_a_frame_that_is_not_newer() {
        let slot = LatestFrame::new();
        slot.put(frame(3));
        // Asking for newer than 5 must not consume (or lose) the frame.
        assert!(slot.take_newer(5, Duration::ZERO).is_none());
        assert_eq!(slot.take_newer(2, Duration::ZERO).map(|f| f.seq), Some(3));
    }

    #[test]
    fn slot_ignores_stale_put_and_keeps_waiting_for_newer() {
        let slot = Arc::new(LatestFrame::new());
        let producer = {
            let slot = Arc::clone(&slot);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                slot.put(frame(3)); // wakes the waiter, but is not newer than 5
                std::thread::sleep(Duration::from_millis(50));
                slot.put(frame(6));
            })
        };
        assert_eq!(
            slot.take_newer(5, Duration::from_secs(5)).map(|f| f.seq),
            Some(6)
        );
        producer.join().expect("producer thread");
    }

    #[test]
    fn slot_put_does_not_wait_for_a_parked_consumer() {
        let slot = Arc::new(LatestFrame::new());
        let consumer = {
            let slot = Arc::clone(&slot);
            // Nothing is ever newer than u64::MAX, so this waits out its whole timeout.
            std::thread::spawn(move || slot.take_newer(u64::MAX, Duration::from_secs(3)))
        };
        std::thread::sleep(Duration::from_millis(50)); // consumer is now waiting in the slot
        let t0 = Instant::now();
        for seq in 1..=10_000 {
            slot.put(frame(seq));
        }
        assert!(
            t0.elapsed() < Duration::from_secs(1),
            "puts blocked behind a waiting consumer: {:?}",
            t0.elapsed()
        );
        assert!(consumer.join().expect("consumer thread").is_none());
    }

    #[test]
    fn slot_concurrent_consumer_sees_increasing_seqs_and_the_last_frame() {
        const LAST: u64 = 5000;
        let slot = Arc::new(LatestFrame::new());
        let producer = {
            let slot = Arc::clone(&slot);
            std::thread::spawn(move || {
                for seq in 1..=LAST {
                    slot.put(frame(seq));
                    if seq % 64 == 0 {
                        std::thread::yield_now();
                    }
                }
            })
        };
        let mut last = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while last < LAST && Instant::now() < deadline {
            if let Some(f) = slot.take_newer(last, Duration::from_millis(50)) {
                assert!(
                    f.seq > last,
                    "stale or repeated frame {} after {last}",
                    f.seq
                );
                last = f.seq;
            }
        }
        producer.join().expect("producer thread");
        // Dropping may skip frames, never the newest one.
        assert_eq!(last, LAST);
    }
}
