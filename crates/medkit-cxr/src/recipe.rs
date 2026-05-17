use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    error::CxrError,
    types::{
        DicomPresentationPolicy, ImageSizePolicy, LabelPolicy, SplitPolicyMetadata,
        TransferSyntaxPolicy,
    },
    util::hash_file,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CxrDicomRecipe {
    pub name: String,
    #[serde(default)]
    pub dicom: RecipeDicomSection,
    #[serde(default)]
    pub presentation: RecipePresentationSection,
    #[serde(default)]
    pub image: RecipeImageSection,
    #[serde(default)]
    pub labels: RecipeLabelsSection,
    #[serde(default)]
    pub split: RecipeSplitSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecipeDicomSection {
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub views: Vec<String>,
    #[serde(default = "default_true")]
    pub require_single_frame: bool,
    #[serde(default = "default_transfer_syntaxes")]
    pub allow_transfer_syntaxes: Vec<String>,
    #[serde(default = "default_unsupported_transfer_syntax")]
    pub unsupported_transfer_syntax: String,
}

impl Default for RecipeDicomSection {
    fn default() -> Self {
        Self {
            modalities: vec!["CR".to_string(), "DX".to_string()],
            views: vec!["PA".to_string(), "AP".to_string()],
            require_single_frame: true,
            allow_transfer_syntaxes: default_transfer_syntaxes(),
            unsupported_transfer_syntax: default_unsupported_transfer_syntax(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecipePresentationSection {
    #[serde(default = "default_true")]
    pub apply_rescale: bool,
    #[serde(default = "default_voi")]
    pub voi: String,
    #[serde(default = "default_true")]
    pub invert_monochrome1: bool,
    #[serde(default = "default_output")]
    pub output: String,
}

impl Default for RecipePresentationSection {
    fn default() -> Self {
        Self {
            apply_rescale: true,
            voi: default_voi(),
            invert_monochrome1: true,
            output: default_output(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecipeImageSection {
    #[serde(default = "default_image_size")]
    pub size: [usize; 2],
    #[serde(default = "default_resize")]
    pub resize: String,
    #[serde(default)]
    pub pad_value: u8,
    #[serde(default = "default_normalize")]
    pub normalize: String,
}

impl Default for RecipeImageSection {
    fn default() -> Self {
        Self {
            size: default_image_size(),
            resize: default_resize(),
            pad_value: 0,
            normalize: default_normalize(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecipeLabelsSection {
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default = "default_ignore")]
    pub uncertain: String,
    #[serde(default = "default_ignore")]
    pub missing: String,
}

impl Default for RecipeLabelsSection {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            uncertain: default_ignore(),
            missing: default_ignore(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecipeSplitSection {
    #[serde(default = "default_split_by")]
    pub by: String,
    #[serde(default = "default_train")]
    pub train: f64,
    #[serde(default = "default_val")]
    pub val: f64,
    #[serde(default = "default_test")]
    pub test: f64,
    #[serde(default)]
    pub stratify: Vec<String>,
    #[serde(default)]
    pub seed: u64,
}

impl Default for RecipeSplitSection {
    fn default() -> Self {
        Self {
            by: default_split_by(),
            train: default_train(),
            val: default_val(),
            test: default_test(),
            stratify: Vec::new(),
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecipeFingerprint {
    pub path: String,
    pub sha256: String,
}

pub fn read_cxr_dicom_recipe(path: &Path) -> Result<CxrDicomRecipe, CxrError> {
    let text = fs::read_to_string(path)?;
    let recipe: CxrDicomRecipe = toml::from_str(&text)?;
    validate_cxr_dicom_recipe(&recipe)?;
    Ok(recipe)
}

pub fn recipe_fingerprint(path: &Path) -> Result<RecipeFingerprint, CxrError> {
    Ok(RecipeFingerprint {
        path: path.display().to_string(),
        sha256: hash_file(path)?,
    })
}

pub fn validate_cxr_dicom_recipe(recipe: &CxrDicomRecipe) -> Result<(), CxrError> {
    if recipe.name.trim().is_empty() {
        return Err(CxrError::Message(
            "recipe name must not be empty".to_string(),
        ));
    }
    if recipe.image.size[0] == 0 || recipe.image.size[1] == 0 {
        return Err(CxrError::Message(
            "recipe image.size values must be greater than zero".to_string(),
        ));
    }
    if recipe.image.size[0] != recipe.image.size[1] {
        return Err(CxrError::Message(
            "only square CXR image.size recipes are supported".to_string(),
        ));
    }
    if recipe.labels.targets.is_empty() {
        return Err(CxrError::Message(
            "recipe labels.targets must contain at least one target".to_string(),
        ));
    }
    for policy in [
        recipe.labels.uncertain.as_str(),
        recipe.labels.missing.as_str(),
    ] {
        if !matches!(
            policy,
            "ignore" | "zero" | "negative" | "one" | "positive" | "fail"
        ) {
            return Err(CxrError::Message(format!(
                "unsupported label policy {policy:?}; expected ignore, zero, negative, one, positive, or fail"
            )));
        }
    }
    if !matches!(
        recipe.dicom.unsupported_transfer_syntax.as_str(),
        "fail" | "warn" | "skip"
    ) {
        return Err(CxrError::Message(format!(
            "unsupported_transfer_syntax must be fail, warn, or skip; got {:?}",
            recipe.dicom.unsupported_transfer_syntax
        )));
    }
    let ratio_sum = recipe.split.train + recipe.split.val + recipe.split.test;
    if !ratio_sum.is_finite() || (ratio_sum - 1.0).abs() > 1.0e-6 {
        return Err(CxrError::Message(format!(
            "split train+val+test must equal 1.0, got {ratio_sum}"
        )));
    }
    if !matches!(recipe.split.by.as_str(), "patient_id" | "patient") {
        return Err(CxrError::Message(format!(
            "only patient-level CXR splits are supported, got {:?}",
            recipe.split.by
        )));
    }
    Ok(())
}

impl CxrDicomRecipe {
    pub fn image_size(&self) -> usize {
        self.image.size[0]
    }

    pub fn label_policy(&self) -> LabelPolicy {
        LabelPolicy {
            positive: "label=1 mask=1".to_string(),
            negative: "label=0 mask=1".to_string(),
            uncertain: self.labels.uncertain.clone(),
            missing: self.labels.missing.clone(),
            loss_mask: format!(
                "uncertain={} missing={}",
                self.labels.uncertain, self.labels.missing
            ),
        }
    }

    pub fn image_size_policy(&self) -> ImageSizePolicy {
        ImageSizePolicy {
            channels: 1,
            height: self.image.size[0],
            width: self.image.size[1],
            dtype: "float32".to_string(),
            transform: format!(
                "resize={} pad_value={} normalize={} output=mono8",
                self.image.resize, self.image.pad_value, self.image.normalize
            ),
        }
    }

    pub fn presentation_policy(&self) -> DicomPresentationPolicy {
        DicomPresentationPolicy {
            apply_rescale: self.presentation.apply_rescale,
            voi: self.presentation.voi.clone(),
            invert_monochrome1: self.presentation.invert_monochrome1,
            output: self.presentation.output.clone(),
            decoder_backend: "medkit-native".to_string(),
            decoder_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn transfer_syntax_policy(&self) -> TransferSyntaxPolicy {
        TransferSyntaxPolicy {
            allow_transfer_syntaxes: self.dicom.allow_transfer_syntaxes.clone(),
            unsupported_transfer_syntax: self.dicom.unsupported_transfer_syntax.clone(),
        }
    }

    pub fn split_policy(&self) -> SplitPolicyMetadata {
        SplitPolicyMetadata {
            by: if self.split.by == "patient" {
                "patient_id".to_string()
            } else {
                self.split.by.clone()
            },
            train: self.split.train,
            val: self.split.val,
            test: self.split.test,
            stratify: self.split.stratify.clone(),
            seed: self.split.seed,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_transfer_syntaxes() -> Vec<String> {
    vec![
        medkit_dicom::IMPLICIT_VR_LITTLE_ENDIAN.to_string(),
        medkit_dicom::EXPLICIT_VR_LITTLE_ENDIAN.to_string(),
        medkit_dicom::EXPLICIT_VR_BIG_ENDIAN.to_string(),
        medkit_dicom::RLE_LOSSLESS.to_string(),
    ]
}

fn default_unsupported_transfer_syntax() -> String {
    "fail".to_string()
}

fn default_voi() -> String {
    "auto".to_string()
}

fn default_output() -> String {
    "mono8".to_string()
}

fn default_image_size() -> [usize; 2] {
    [512, 512]
}

fn default_resize() -> String {
    "fit".to_string()
}

fn default_normalize() -> String {
    "train_split_mean_std".to_string()
}

fn default_ignore() -> String {
    "ignore".to_string()
}

fn default_split_by() -> String {
    "patient_id".to_string()
}

fn default_train() -> f64 {
    0.8
}

fn default_val() -> f64 {
    0.1
}

fn default_test() -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    fn recipe_defaults_fingerprint_and_policy_metadata_are_stable() {
        let root = unique_test_dir();
        fs::create_dir_all(&root).unwrap();
        let path = root.join("recipe.toml");
        fs::write(
            &path,
            r#"
name = "defaults"

[labels]
targets = ["Pneumonia"]

[split]
by = "patient"
train = 1.0
val = 0.0
test = 0.0
"#,
        )
        .unwrap();

        let recipe = read_cxr_dicom_recipe(&path).unwrap();
        assert_eq!(recipe.image.size, [512, 512]);
        assert_eq!(recipe.image.normalize, "train_split_mean_std");
        assert!(recipe
            .dicom
            .allow_transfer_syntaxes
            .contains(&medkit_dicom::RLE_LOSSLESS.to_string()));
        assert_eq!(recipe.image_size(), 512);
        assert_eq!(recipe.label_policy().missing, "ignore");
        assert!(recipe.image_size_policy().transform.contains("resize=fit"));
        assert_eq!(
            recipe.presentation_policy().decoder_backend,
            "medkit-native"
        );
        assert_eq!(
            recipe.transfer_syntax_policy().unsupported_transfer_syntax,
            "fail"
        );
        assert_eq!(recipe.split_policy().by, "patient_id");

        let fingerprint = recipe_fingerprint(&path).unwrap();
        assert_eq!(fingerprint.path, path.display().to_string());
        assert_eq!(fingerprint.sha256.len(), 64);
    }

    #[test]
    fn recipe_validation_rejects_invalid_policy_shapes() {
        let mut recipe = valid_recipe();
        recipe.name = " ".to_string();
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("name"));

        recipe = valid_recipe();
        recipe.image.size = [0, 512];
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("image.size"));

        recipe = valid_recipe();
        recipe.image.size = [256, 512];
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("square"));

        recipe = valid_recipe();
        recipe.labels.targets.clear();
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("labels.targets"));

        recipe = valid_recipe();
        recipe.labels.uncertain = "maybe".to_string();
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("unsupported label policy"));

        recipe = valid_recipe();
        recipe.dicom.unsupported_transfer_syntax = "defer".to_string();
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("unsupported_transfer_syntax"));

        recipe = valid_recipe();
        recipe.split.train = 0.5;
        recipe.split.val = 0.5;
        recipe.split.test = 0.5;
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("must equal 1.0"));

        recipe = valid_recipe();
        recipe.split.by = "study_id".to_string();
        assert!(validate_cxr_dicom_recipe(&recipe)
            .unwrap_err()
            .to_string()
            .contains("patient-level"));
    }

    fn valid_recipe() -> CxrDicomRecipe {
        CxrDicomRecipe {
            name: "valid".to_string(),
            dicom: RecipeDicomSection::default(),
            presentation: RecipePresentationSection::default(),
            image: RecipeImageSection::default(),
            labels: RecipeLabelsSection {
                targets: vec!["Pneumonia".to_string()],
                ..RecipeLabelsSection::default()
            },
            split: RecipeSplitSection::default(),
        }
    }

    fn unique_test_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "medkit-cxr-recipe-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }
}
