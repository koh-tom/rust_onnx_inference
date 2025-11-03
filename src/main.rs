use ndarray::{Array, CowArray};
use ort::{Environment, GraphOptimizationLevel, LoggingLevel, OrtResult, SessionBuilder, Value};
use std::sync::Arc;

fn main() -> OrtResult<()> {
    println!("📦 モデルの読み込み中...");

    // 1. onnxの環境を作成
    let environment = Arc::new(
        Environment::builder()
            .with_name("rust_onnx_infer")
            .with_log_level(LoggingLevel::Warning)
            .build()?,
    );

    // 2. セッションを作成し、モデルを読み込む
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("model/yolov5s.onnx")?; // モデルパスを指定

    println!("✅ モデル読み込み完了!");

    // 3. 入力データを準備
    // ダミーデータを作成
    // バッチサイズ1、チャンネル3、幅320、高さ320の例
    let array: Array<f32, _> = Array::zeros((1, 3, 320, 320)).into_dyn(); // 入力のshapeを指定
    let cow_array = CowArray::from(&array);
    let input_tensor = Value::from_array(session.allocator(), &cow_array)?;

    // 4. 推論を実行
    let outputs: Vec<Value> = session.run(vec![input_tensor])?;

    // 5. 出力を処理
    let output_tensor = &outputs[0];
    let output_tensor = output_tensor.try_extract::<f32>()?;
    println!("📊 出力shape: {:?}", output_tensor.view().shape());

    Ok(())
}
