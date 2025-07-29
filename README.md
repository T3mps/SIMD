# SIMD Abstraction Library

A modern C++20 header-only library for explicit, type-safe SIMD operations across x86/x64 and ARM64 platforms.

## Features

- **Explicit width control** - No surprises. You specify whether you want 128-bit or 256-bit operations.
- **Zero dependencies** - Single header, copy and use.
- **Cross-platform** - Unified API for SSE2/AVX2 (x86/x64) and NEON (ARM64).
- **Type-safe** - Return types are predictable and consistent across platforms.
- **Production-tested** - Optimized implementations with proper alignment handling.

## Requirements

- C++20 compiler
- x86/x64: SSE2 minimum, AVX2 for 256-bit operations  
- ARM: ARMv8 (64-bit) with NEON

## Performance Notes

- Always ensure proper alignment (16 bytes for 128-bit, 32 bytes for 256-bit)
- Batch operations provide better throughput for processing multiple vectors
- Prefetch operations are available for memory-bound workloads
- Fallback implementations are optimized for platforms without native support
