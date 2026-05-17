use crate::{pixel::present_dicom_pixels, DicomError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderOptions {
    pub width: usize,
    pub include_metadata: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 80,
            include_metadata: true,
        }
    }
}

pub fn render_unicode(
    path: impl AsRef<std::path::Path>,
    options: &RenderOptions,
) -> Result<String> {
    let image = present_dicom_pixels(path)?;
    if options.width == 0 {
        return Err(DicomError::InvalidInput(
            "render width must be greater than zero".to_string(),
        ));
    }
    let out_width = options.width.min(image.width.max(1));
    let scale_x = image.width as f32 / out_width as f32;
    let out_height = ((image.height as f32 / scale_x) * 0.5).ceil().max(1.0) as usize;
    let scale_y = image.height as f32 / out_height as f32;
    let palette = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    let mut out = String::new();
    if options.include_metadata {
        out.push_str(&format!(
            "DICOM {}x{} {}\n",
            image.width, image.height, image.explanation.photometric_interpretation
        ));
        out.push_str(&format!(
            "transfer syntax: {}\n",
            image.explanation.transfer_syntax_uid
        ));
    }
    for y in 0..out_height {
        let src_y = ((y as f32 + 0.5) * scale_y)
            .floor()
            .min((image.height - 1) as f32) as usize;
        for x in 0..out_width {
            let src_x = ((x as f32 + 0.5) * scale_x)
                .floor()
                .min((image.width - 1) as f32) as usize;
            let value = image.pixels[src_y * image.width + src_x] as usize;
            let bucket = value * (palette.len() - 1) / 255;
            out.push(palette[bucket]);
        }
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_render_options_are_terminal_friendly() {
        let options = RenderOptions::default();
        assert_eq!(options.width, 80);
        assert!(options.include_metadata);
    }
}
