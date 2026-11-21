//! Vision error type.

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    /// Model file or model output not usable (wrong shape, unexpected graph, ...).
    #[error("model error: {0}")]
    Model(String),
    #[error("onnx runtime error: {0}")]
    Ort(String),
    /// Caller-supplied input not usable (bad frame, degenerate box, ...).
    #[error("invalid input: {0}")]
    Input(String),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

/// Generic over the recoverable payload (`ort::Error<SessionBuilder>` from builder calls).
impl<R> From<ort::Error<R>> for VisionError {
    fn from(e: ort::Error<R>) -> Self {
        VisionError::Ort(e.to_string())
    }
}
