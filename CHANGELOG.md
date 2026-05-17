# Changelog

## 0.1.0 - 2026-05-17

Initial release of the Python package.

### Included

- Rust-backed dataset validation and deterministic preprocessing cache tooling.
- Multi-channel 3D cache preparation and channel-aware patch sampling.
- CXR cache readers and Python dataset/DataLoader helpers.
- Native DICOM pixel presentation for uncompressed little/big endian, RLE Lossless, and JPEG Baseline.
- Optional `dicom-rs-codecs` feature for the DICOM-rs pixel backend.

### Packaging Notes

- NumPy is a required runtime dependency.
- Torch is optional and available as `medkit-rs[torch]`.
- Free-threaded CPython builds are not supported for this release.
- JPEG-LS and JPEG 2000 codec stacks are not enabled in the default package.
