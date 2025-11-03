use opencv::{highgui, imgcodecs, prelude::*, videoio};

use crate::AppError;

pub fn get_camera_frame(_input_width: i32, _input_height: i32) -> Result<(), AppError> {
    println!("📷 カメラを起動中...");

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("カメラが開けない！");
    }

    let window = "YOLOv5 ONNX";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
    println!("✅ カメラ起動完了! 'esc'キーで終了します。");

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.empty() {
            break;
        }

        // 映像表示
        highgui::imshow(window, &frame)?;
        let key = highgui::wait_key(1)?;
        let frame_count = 0;

        if key == 27 {
            break;
        } else if key == 115 {
            let filename = format!("img/frame_{}.png", frame_count);
            imgcodecs::imwrite(&filename, &frame, &opencv::types::VectorOfi32::new())?;
            println!("✅ フレームを保存！: {}", filename);
        }
    }

    Ok(())
}
