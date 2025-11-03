use crate::AppError;
use ort::{Environment, GraphOptimizationLevel, LoggingLevel, Session, SessionBuilder, Value};
use std::sync::Arc;

// --- セッションのセットアップ関数 ---
pub fn setup_session() -> Result<Session, AppError> {
    println!("📦 モデルの読み込み中...");

    // onnxの環境を作成
    let environment = Arc::new(
        Environment::builder()
            .with_name("rust_onnx_infer")
            .with_log_level(LoggingLevel::Warning)
            .build()?,
    );

    // セッションを作成し、モデルを読み込む
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("model/yolov5s.onnx")?;

    println!("✅ モデル読み込み完了!");
    Ok(session)
}

// --- 推論を実行する関数 ---
pub fn run_inference(session: &Session, input_tensor: Value) -> Result<(), AppError> {
    println!("🚀 推論を実行中...");
    let outputs: Vec<Value> = session.run(vec![input_tensor])?;
    println!("✅ 推論完了!");

    // 出力を処理
    let output_tensor = &outputs[0];
    let output_tensor = output_tensor.try_extract::<f32>()?;
    println!("📊 出力shape: {:?}", output_tensor.view().shape());

    Ok(())
}
