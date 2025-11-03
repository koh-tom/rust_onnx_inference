use ndarray::CowArray;
use ort::Value;
use thiserror::Error;

mod camera;
mod image_processing;
mod inference;

// アプリケーションエラーの定義
#[derive(Error, Debug)]
pub enum AppError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::OrtError),
    #[error("OpenCV error: {0}")]
    OpenCv(#[from] opencv::Error),
    #[error("Image not found: {0}")]
    ImageNotFound(String),
}

fn main() -> Result<(), AppError> {
    // カメラからフレームを取得
    camera::get_camera_frame(640, 640)?;

    let session = inference::setup_session()?;

    let image_path = "img/cat.jpg";
    let input_width = 640;
    let input_height = 640;

    // 画像の前処理
    let preprocessed_image =
        image_processing::preprocess_image(image_path, input_width, input_height)?;

    // ndarrayをONNX RuntimeのValueに変換
    let cow_array = CowArray::from(preprocessed_image);

    // 入力テンソルを作成
    let input_tensor = Value::from_array(session.allocator(), &cow_array)?;

    // 推論を実行
    inference::run_inference(&session, input_tensor)?;

    Ok(())
}
