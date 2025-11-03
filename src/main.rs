use ndarray::{Array, CowArray};
use opencv::{core, imgcodecs, imgproc, prelude::*};
use ort::{Environment, GraphOptimizationLevel, LoggingLevel, SessionBuilder, Value};
use std::sync::Arc;
use thiserror::Error;

mod camera;

// アプリケーションエラーの定義
#[derive(Error, Debug)]

enum AppError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::OrtError),
    #[error("OpenCV error: {0}")]
    OpenCv(#[from] opencv::Error),
    #[error("Image not found: {0}")]
    ImageNotFound(String),
}

fn main() -> Result<(), AppError> {
    camera::get_camera_frame(640, 640)?;
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
        .with_model_from_file("model/yolov5s.onnx")?;

    println!("✅ モデル読み込み完了!");

    // 3. 画像を読み込み、前処理を行う
    println!("🖼️ 画像を読み込み、前処理中...");
    let image_path = "img/cat.jpg"; // 画像パスを指定
    let original_image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    if original_image.empty() {
        return Err(AppError::ImageNotFound(image_path.to_string()));
    }

    // YOLOv5の入力サイズ
    let input_width = 640;
    let input_height = 640;

    // 画像のサイズ変換
    let mut resized_image = Mat::default();
    imgproc::resize(
        &original_image,
        &mut resized_image,
        core::Size::new(input_width, input_height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // BGR to RGB
    let mut rgb_image = Mat::default();
    imgproc::cvt_color(&resized_image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;

    // 0-1に正規化
    let mut normalized_image = Mat::default();
    rgb_image.convert_to(&mut normalized_image, core::CV_32F, 1.0 / 255.0, 0.0)?;

    // HWC to CHW
    let mut channels = core::Vector::<Mat>::new();
    core::split(&normalized_image, &mut channels)?;
    let mut chw_image_data: Vec<f32> = Vec::new();
    for i in 0..channels.len() {
        let channel = channels.get(i)?;
        let data = channel.data_typed::<f32>()?;
        chw_image_data.extend_from_slice(data);
    }

    // ndarrayに変換
    let array = Array::from_shape_vec(
        (1, 3, input_height as usize, input_width as usize),
        chw_image_data,
    )
    .unwrap()
    .into_dyn();

    let cow_array = CowArray::from(&array);
    let input_tensor = Value::from_array(session.allocator(), &cow_array)?;

    println!("✅ 前処理完了!");

    // 4. 推論を実行
    println!("🚀 推論を実行中...");
    let outputs: Vec<Value> = session.run(vec![input_tensor])?;
    println!("✅ 推論完了!");

    // 5. 出力を処理
    let output_tensor = &outputs[0];
    let output_tensor = output_tensor.try_extract::<f32>()?;
    println!("📊 出力shape: {:?}", output_tensor.view().shape());

    Ok(())
}
