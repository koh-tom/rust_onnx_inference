use crate::AppError;
use ndarray::{Array, IxDyn};
use opencv::{core, imgcodecs, imgproc, prelude::*};

// --- 画像の前処理関数 ---
pub fn preprocess_image(
    image_path: &str,
    input_width: i32,
    input_height: i32,
) -> Result<Array<f32, IxDyn>, AppError> {
    println!("🖼️ 画像を読み込み、前処理中...");
    let original_image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    if original_image.empty() {
        return Err(AppError::ImageNotFound(image_path.to_string()));
    }

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

    println!("✅ 前処理完了!");
    Ok(array)
}
