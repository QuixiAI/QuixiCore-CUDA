# Changelog

All notable QuixiCore CUDA changes should be recorded here.

## Unreleased

- quant: cp.async double-buffered y tiles in the segmented MXFP4 MoE
  pipeline (span-parity buffers, 4-byte cp.async.ca, dynamic smem with
  opt-in) -- 26.9 -> 20.2 ms/iter (-25%) at the DSV4 TP4 seg shape on
  A100; bit-exact vs the synchronous loaders (SlimServe port).
- quant: fused Q4_K (GGUF type 12) MoE decode pair -- warp-per-row gate/up
  GEMV with fused SwiGLU + route weight + Q8_1 emission and a Q8xQ4_K
  weighted down sum over raw block_q4_K rows, with a standalone
  correctness harness (SlimServe port).
- Added QuixiCore-standard repository structure documentation.
- Added QuixiCore metadata manifests for backend identity, kernel family
  coverage, and quant format coverage.
- Added standardized contributor, security, changelog, formatting, and script
  entrypoint files.
