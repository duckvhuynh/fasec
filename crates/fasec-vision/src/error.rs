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

impl From<ort::Error> for VisionError {
    fn from(e: ort::Error) -> Self {
        VisionError::Ort(e.to_string())
    }
}
