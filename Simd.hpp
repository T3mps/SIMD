#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cassert>

// Self-contained SIMD abstraction layer with explicit width control
// 
// IMPORTANT: This library requires explicit width specification to ensure:
// - Type safety: Return types are predictable and consistent
// - Algorithm correctness: Loop bounds and pointer advancement are explicit
// - Memory alignment: Each width has specific alignment requirements
//   - 128-bit operations require 16-byte aligned data
//   - 256-bit operations require 32-byte aligned data
//
// Platform Requirements:
// - x86/x64: SSE2 minimum, AVX2 for 256-bit operations
// - ARM: ARMv8 (64-bit) with NEON. ARMv7 is NOT supported due to use of
//        vaddv_u8 and other ARMv8-specific instructions
//
// Usage:
// - Query capabilities: Simd::Capabilities::HasWidth<Width>()
// - Explicit 128-bit: Simd::Ops::MatchByteMask<Simd::Width128>(data, value)
// - Explicit 256-bit: Simd::Ops::MatchByteMask<Simd::Width256>(data, value)
//
// Always choose width based on your algorithm's requirements, not just
// "best available". Consider cache utilization, register pressure, and
// algorithm structure when selecting SIMD width.

// ============== Platform Detection ==============

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
    #define SIMD_ARCH_X64 1
#elif defined(__i386__) || defined(_M_IX86)
    #define SIMD_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define SIMD_ARCH_ARM64 1
#elif defined(__arm__) || defined(_M_ARM)
    #define SIMD_ARCH_ARM32 1
#endif

// Compiler detection
#if defined(_MSC_VER)
    #define SIMD_COMPILER_MSVC 1
#elif defined(__clang__)
    #define SIMD_COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define SIMD_COMPILER_GCC 1
#endif

// Feature detection macros
#ifdef __has_builtin
    #define SIMD_HAS_BUILTIN(x) __has_builtin(x)
#else
    #define SIMD_HAS_BUILTIN(x) 0
#endif

// SIMD capability detection
#if defined(SIMD_ARCH_X64) || defined(SIMD_ARCH_X86)
    #ifdef __SSE2__
        #define SIMD_HAS_SSE2 1
    #endif
    #ifdef __SSE4_2__
        #define SIMD_HAS_SSE42 1
    #endif
    #ifdef __AVX__
        #define SIMD_HAS_AVX 1
    #endif
    #ifdef __AVX2__
        #define SIMD_HAS_AVX2 1
    #endif
#elif defined(SIMD_ARCH_ARM64)
    // ARMv8 always has NEON
    #define SIMD_HAS_NEON 1
    #if defined(__ARM_FEATURE_CRC32)
        #define SIMD_HAS_ARM_CRC32 1
    #endif
#endif

// ============== Utility Macros ==============

// Forceinline
#if defined(SIMD_COMPILER_MSVC)
    #define SIMD_FORCEINLINE __forceinline
#elif defined(SIMD_COMPILER_GCC) || defined(SIMD_COMPILER_CLANG)
    #define SIMD_FORCEINLINE inline __attribute__((always_inline))
#else
    #define SIMD_FORCEINLINE inline
#endif

// Debug assertions
#ifdef NDEBUG
    #define SIMD_ASSERT(condition, message) ((void)0)
#else
    #define SIMD_ASSERT(condition, message) assert((condition) && (message))
#endif

// ============== Includes ==============

#if defined(SIMD_ARCH_X64) || defined(SIMD_ARCH_X86)
    #if defined(SIMD_COMPILER_MSVC)
        #include <intrin.h>
        #include <nmmintrin.h>  // SSE4.2
        #if defined(SIMD_HAS_AVX) || defined(SIMD_HAS_AVX2)
            #include <immintrin.h>  // AVX/AVX2
        #endif
    #else
        #include <x86intrin.h>
    #endif
#elif defined(SIMD_ARCH_ARM64)
    #include <arm_neon.h>
#endif

namespace Simd
{
    // Width tags for explicit control
    struct Width128 {};  // 16 bytes
    struct Width256 {};  // 32 bytes
    
    // Concept for valid SIMD widths
    template<typename T>
    concept SimdWidth = std::is_same_v<T, Width128> || std::is_same_v<T, Width256>;
    
    // Width traits
    template<typename Width>
    struct WidthTraits;
    
    template<>
    struct WidthTraits<Width128>
    {
        static constexpr size_t bytes = 16;
        using MaskType = uint16_t;
    };
    
    template<>
    struct WidthTraits<Width256>
    {
        static constexpr size_t bytes = 32;
        using MaskType = uint32_t;
    };

    // Alignment utilities
    template<SimdWidth Width>
    inline constexpr size_t AlignmentV = WidthTraits<Width>::bytes;

    template<SimdWidth Width>
    [[nodiscard]] SIMD_FORCEINLINE bool IsAligned(const void* ptr) noexcept
    {
        return (reinterpret_cast<uintptr_t>(ptr) & (AlignmentV<Width> - 1)) == 0;
    }

    // Capability detection
    namespace Capabilities
    {
        // Check if a specific width is supported with hardware acceleration
        template<SimdWidth Width>
        [[nodiscard]] inline constexpr bool HasWidth() noexcept
        {
            if constexpr (std::is_same_v<Width, Width128>)
            {
#if defined(SIMD_HAS_SSE2) || defined(SIMD_HAS_NEON)
                return true;
#else
                return false;
#endif
            }
            else if constexpr (std::is_same_v<Width, Width256>)
            {
#if defined(SIMD_HAS_AVX2)
                return true;
#else
                return false;
#endif
            }
            return false;
        }

        // Check if a specific width has optimized fallback
        template<SimdWidth Width>
        [[nodiscard]] inline constexpr bool HasFallback() noexcept
        {
            if constexpr (std::is_same_v<Width, Width256>)
            {
                // 256-bit can fall back to 2x128 if 128-bit is available
                return HasWidth<Width128>();
            }
            return true; // Scalar fallback always available
        }

        // Get the best width for a given data size
        // This is for informational purposes only - users must still specify width explicitly
        [[nodiscard]] inline constexpr size_t SuggestedWidthForSize(size_t data_size) noexcept
        {
            // For small data, 128-bit might be more efficient due to lower latency
            if (data_size <= 64)
                return 16;
            
            // For larger data, use wider vectors if available
#if defined(SIMD_HAS_AVX2)
            return 32;
#else
            return 16;
#endif
        }
    }

    namespace Ops
    {
        // ============== 128-bit implementations ==============
        namespace Detail128
        {
            SIMD_FORCEINLINE uint16_t MatchByteMask_SSE(const void* data, uint8_t value) noexcept
            {
#if defined(SIMD_HAS_SSE2)
                const __m128i group = _mm_load_si128(static_cast<const __m128i*>(data));
                const __m128i match = _mm_set1_epi8(static_cast<char>(value));
                const __m128i eq = _mm_cmpeq_epi8(group, match);
                return static_cast<uint16_t>(_mm_movemask_epi8(eq));
#else
                return 0; // Should not reach here
#endif
            }

            SIMD_FORCEINLINE uint16_t MatchByteMask_NEON(const void* data, uint8_t value) noexcept
            {
#if defined(SIMD_HAS_NEON)
                const uint8x16_t group = vld1q_u8(static_cast<const uint8_t*>(data));
                const uint8x16_t match = vdupq_n_u8(value);
                const uint8x16_t eq = vceqq_u8(group, match);

                const uint8x16_t bit_mask = {
                    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
                };
                const uint8x16_t masked = vandq_u8(eq, bit_mask);
                const uint8x8_t low = vget_low_u8(masked);
                const uint8x8_t high = vget_high_u8(masked);

                uint8_t low_mask = vaddv_u8(low);
                uint8_t high_mask = vaddv_u8(high);
                return (static_cast<uint16_t>(high_mask) << 8) | low_mask;
#else
                return 0; // Should not reach here
#endif
            }

            SIMD_FORCEINLINE uint16_t MatchByteMask_Scalar(const void* data, uint8_t value) noexcept
            {
                uint16_t mask = 0;
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                for (int i = 0; i < 16; ++i)
                {
                    if (bytes[i] == value)
                    {
                        mask |= (1u << i);
                    }
                }
                return mask;
            }

            SIMD_FORCEINLINE uint16_t MatchEitherByteMask_SSE(const void* data, uint8_t val1, uint8_t val2) noexcept
            {
#if defined(SIMD_HAS_SSE2)
                const __m128i group = _mm_load_si128(static_cast<const __m128i*>(data));
                const __m128i match1 = _mm_set1_epi8(static_cast<char>(val1));
                const __m128i match2 = _mm_set1_epi8(static_cast<char>(val2));
                const __m128i eq1 = _mm_cmpeq_epi8(group, match1);
                const __m128i eq2 = _mm_cmpeq_epi8(group, match2);
                const __m128i combined = _mm_or_si128(eq1, eq2);
                return static_cast<uint16_t>(_mm_movemask_epi8(combined));
#else
                return 0;
#endif
            }

            SIMD_FORCEINLINE uint16_t MatchEitherByteMask_NEON(const void* data, uint8_t val1, uint8_t val2) noexcept
            {
#if defined(SIMD_HAS_NEON)
                const uint8x16_t group = vld1q_u8(static_cast<const uint8_t*>(data));
                const uint8x16_t match1 = vdupq_n_u8(val1);
                const uint8x16_t match2 = vdupq_n_u8(val2);
                const uint8x16_t eq1 = vceqq_u8(group, match1);
                const uint8x16_t eq2 = vceqq_u8(group, match2);
                const uint8x16_t combined = vorrq_u8(eq1, eq2);

                const uint8x16_t bit_mask = {
                    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
                };
                const uint8x16_t masked = vandq_u8(combined, bit_mask);
                const uint8x8_t low = vget_low_u8(masked);
                const uint8x8_t high = vget_high_u8(masked);

                uint8_t low_mask = vaddv_u8(low);
                uint8_t high_mask = vaddv_u8(high);
                return (static_cast<uint16_t>(high_mask) << 8) | low_mask;
#else
                return 0;
#endif
            }

            SIMD_FORCEINLINE uint16_t MatchEitherByteMask_Scalar(const void* data, uint8_t val1, uint8_t val2) noexcept
            {
                uint16_t mask = 0;
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                for (int i = 0; i < 16; ++i)
                {
                    if (bytes[i] == val1 || bytes[i] == val2)
                    {
                        mask |= (1u << i);
                    }
                }
                return mask;
            }
        }

        // ============== 256-bit implementations ==============
        // Note: Fallback implementations are optimized for instruction scheduling:
        // - Both 128-bit loads are issued before any comparisons
        // - Match values are broadcast once and reused
        // - Operations are interleaved to hide latency
        namespace Detail256
        {
            SIMD_FORCEINLINE uint32_t MatchByteMask_AVX(const void* data, uint8_t value) noexcept
            {
#if defined(SIMD_HAS_AVX2)
                const __m256i group = _mm256_load_si256(static_cast<const __m256i*>(data));
                const __m256i match = _mm256_set1_epi8(static_cast<char>(value));
                const __m256i eq = _mm256_cmpeq_epi8(group, match);
                return static_cast<uint32_t>(_mm256_movemask_epi8(eq));
#else
                return 0;
#endif
            }

            SIMD_FORCEINLINE uint32_t MatchByteMask_Fallback(const void* data, uint8_t value) noexcept
            {
#if defined(SIMD_HAS_SSE2)
                // Optimized 2x128 with better scheduling
                const uint8_t* ptr = static_cast<const uint8_t*>(data);
                const __m128i* ptr_low = reinterpret_cast<const __m128i*>(ptr);
                const __m128i* ptr_high = reinterpret_cast<const __m128i*>(ptr + 16);
                
                // Load both halves first to hide latency
                const __m128i group_low = _mm_load_si128(ptr_low);
                const __m128i group_high = _mm_load_si128(ptr_high);
                
                // Broadcast match value once
                const __m128i match = _mm_set1_epi8(static_cast<char>(value));
                
                // Compare both halves
                const __m128i eq_low = _mm_cmpeq_epi8(group_low, match);
                const __m128i eq_high = _mm_cmpeq_epi8(group_high, match);
                
                // Extract masks
                const uint16_t mask_low = static_cast<uint16_t>(_mm_movemask_epi8(eq_low));
                const uint16_t mask_high = static_cast<uint16_t>(_mm_movemask_epi8(eq_high));
                
                return (static_cast<uint32_t>(mask_high) << 16) | mask_low;
#else
                // Fallback to function calls
                auto mask_low = Detail128::MatchByteMask_Scalar(data, value);
                auto mask_high = Detail128::MatchByteMask_Scalar(static_cast<const uint8_t*>(data) + 16, value);
                return (static_cast<uint32_t>(mask_high) << 16) | mask_low;
#endif
            }

            SIMD_FORCEINLINE uint32_t MatchEitherByteMask_AVX(const void* data, uint8_t val1, uint8_t val2) noexcept
            {
#if defined(SIMD_HAS_AVX2)
                const __m256i group = _mm256_load_si256(static_cast<const __m256i*>(data));
                const __m256i match1 = _mm256_set1_epi8(static_cast<char>(val1));
                const __m256i match2 = _mm256_set1_epi8(static_cast<char>(val2));
                const __m256i eq1 = _mm256_cmpeq_epi8(group, match1);
                const __m256i eq2 = _mm256_cmpeq_epi8(group, match2);
                const __m256i combined = _mm256_or_si256(eq1, eq2);
                return static_cast<uint32_t>(_mm256_movemask_epi8(combined));
#else
                return 0;
#endif
            }

            SIMD_FORCEINLINE uint32_t MatchEitherByteMask_Fallback(const void* data, uint8_t val1, uint8_t val2) noexcept
            {
#if defined(SIMD_HAS_SSE2)
                // Optimized 2x128 with better scheduling
                const uint8_t* ptr = static_cast<const uint8_t*>(data);
                const __m128i* ptr_low = reinterpret_cast<const __m128i*>(ptr);
                const __m128i* ptr_high = reinterpret_cast<const __m128i*>(ptr + 16);
                
                // Load both halves first
                const __m128i group_low = _mm_load_si128(ptr_low);
                const __m128i group_high = _mm_load_si128(ptr_high);
                
                // Broadcast match values once
                const __m128i match1 = _mm_set1_epi8(static_cast<char>(val1));
                const __m128i match2 = _mm_set1_epi8(static_cast<char>(val2));
                
                // Compare both values on both halves
                const __m128i eq1_low = _mm_cmpeq_epi8(group_low, match1);
                const __m128i eq2_low = _mm_cmpeq_epi8(group_low, match2);
                const __m128i eq1_high = _mm_cmpeq_epi8(group_high, match1);
                const __m128i eq2_high = _mm_cmpeq_epi8(group_high, match2);
                
                // Combine results
                const __m128i combined_low = _mm_or_si128(eq1_low, eq2_low);
                const __m128i combined_high = _mm_or_si128(eq1_high, eq2_high);
                
                // Extract masks
                const uint16_t mask_low = static_cast<uint16_t>(_mm_movemask_epi8(combined_low));
                const uint16_t mask_high = static_cast<uint16_t>(_mm_movemask_epi8(combined_high));
                
                return (static_cast<uint32_t>(mask_high) << 16) | mask_low;
#else
                auto mask_low = Detail128::MatchEitherByteMask_Scalar(data, val1, val2);
                auto mask_high = Detail128::MatchEitherByteMask_Scalar(static_cast<const uint8_t*>(data) + 16, val1, val2);
                return (static_cast<uint32_t>(mask_high) << 16) | mask_low;
#endif
            }
        }

        // ============== Public API ==============
        // IMPORTANT: All operations require explicit width specification
        // This ensures type safety and predictable behavior across platforms

        // Explicit width control
        template<SimdWidth Width>
        SIMD_FORCEINLINE auto MatchByteMask(const void* data, uint8_t value) noexcept -> typename WidthTraits<Width>::MaskType
        {
            SIMD_ASSERT(IsAligned<Width>(data), "Data must be aligned for SIMD operations");
            
            if constexpr (std::is_same_v<Width, Width128>)
            {
#if defined(SIMD_HAS_SSE2)
                return Detail128::MatchByteMask_SSE(data, value);
#elif defined(SIMD_HAS_NEON)
                return Detail128::MatchByteMask_NEON(data, value);
#else
                return Detail128::MatchByteMask_Scalar(data, value);
#endif
            }
            else if constexpr (std::is_same_v<Width, Width256>)
            {
#if defined(SIMD_HAS_AVX2)
                return Detail256::MatchByteMask_AVX(data, value);
#elif defined(SIMD_HAS_SSE2)
                return Detail256::MatchByteMask_Fallback(data, value);
#else
                // Scalar fallback for 256-bit
                uint32_t mask = 0;
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                for (int i = 0; i < 32; ++i)
                {
                    if (bytes[i] == value)
                    {
                        mask |= (1u << i);
                    }
                }
                return mask;
#endif
            }
        }

        template<SimdWidth Width>
        SIMD_FORCEINLINE auto MatchEitherByteMask(const void* data, uint8_t val1, uint8_t val2) noexcept -> typename WidthTraits<Width>::MaskType
        {
            SIMD_ASSERT(IsAligned<Width>(data), "Data must be aligned for SIMD operations");
            
            if constexpr (std::is_same_v<Width, Width128>)
            {
#if defined(SIMD_HAS_SSE2)
                return Detail128::MatchEitherByteMask_SSE(data, val1, val2);
#elif defined(SIMD_HAS_NEON)
                return Detail128::MatchEitherByteMask_NEON(data, val1, val2);
#else
                return Detail128::MatchEitherByteMask_Scalar(data, val1, val2);
#endif
            }
            else if constexpr (std::is_same_v<Width, Width256>)
            {
#if defined(SIMD_HAS_AVX2)
                return Detail256::MatchEitherByteMask_AVX(data, val1, val2);
#elif defined(SIMD_HAS_SSE2)
                return Detail256::MatchEitherByteMask_Fallback(data, val1, val2);
#else
                // Scalar fallback for 256-bit
                uint32_t mask = 0;
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                for (int i = 0; i < 32; ++i)
                {
                    if (bytes[i] == val1 || bytes[i] == val2)
                    {
                        mask |= (1u << i);
                    }
                }
                return mask;
#endif
            }
        }

        // Population count (number of set bits)
        template<typename MaskType>
        SIMD_FORCEINLINE int PopCount(MaskType mask) noexcept
        {
#if defined(SIMD_COMPILER_MSVC)
            if constexpr (sizeof(MaskType) <= 4)
            {
                return static_cast<int>(__popcnt(static_cast<unsigned>(mask)));
            }
            else
            {
                #ifdef _WIN64
                return static_cast<int>(__popcnt64(static_cast<unsigned long long>(mask)));
                #else
                return __popcnt(static_cast<unsigned>(mask)) + 
                       __popcnt(static_cast<unsigned>(mask >> 32));
                #endif
            }
#elif SIMD_HAS_BUILTIN(__builtin_popcount) || SIMD_HAS_BUILTIN(__builtin_popcountll)
            if constexpr (sizeof(MaskType) <= 4)
            {
                return __builtin_popcount(static_cast<unsigned>(mask));
            }
            else
            {
                return __builtin_popcountll(static_cast<unsigned long long>(mask));
            }
#else
            // Brian Kernighan's algorithm
            int count = 0;
            while (mask)
            {
                mask &= mask - 1;
                count++;
            }
            return count;
#endif
        }

        // Find first set bit (1-indexed, returns 0 if no bits set)
        template<typename MaskType>
        SIMD_FORCEINLINE int FindFirstSet(MaskType mask) noexcept
        {
            if (!mask) return 0;
            return CountTrailingZeros(mask) + 1;
        }

        // Find last set bit (1-indexed, returns 0 if no bits set)
        template<typename MaskType>
        SIMD_FORCEINLINE int FindLastSet(MaskType mask) noexcept
        {
            if (!mask) return 0;
            
#if defined(SIMD_COMPILER_MSVC)
            unsigned long idx;
            if constexpr (sizeof(MaskType) <= 4)
            {
                _BitScanReverse(&idx, static_cast<unsigned long>(mask));
                return static_cast<int>(idx) + 1;
            }
            else
            {
                #ifdef _WIN64
                _BitScanReverse64(&idx, static_cast<unsigned long long>(mask));
                return static_cast<int>(idx) + 1;
                #else
                if (mask >> 32)
                {
                    _BitScanReverse(&idx, static_cast<unsigned>(mask >> 32));
                    return static_cast<int>(idx) + 33;
                }
                else
                {
                    _BitScanReverse(&idx, static_cast<unsigned>(mask));
                    return static_cast<int>(idx) + 1;
                }
                #endif
            }
#elif SIMD_HAS_BUILTIN(__builtin_clz) || SIMD_HAS_BUILTIN(__builtin_clzll)
            if constexpr (sizeof(MaskType) <= 4)
            {
                return 32 - __builtin_clz(static_cast<unsigned>(mask));
            }
            else
            {
                return 64 - __builtin_clzll(static_cast<unsigned long long>(mask));
            }
#else
            // Fallback
            int pos = 0;
            while (mask)
            {
                pos++;
                mask >>= 1;
            }
            return pos;
#endif
        }

        // Count trailing zeros - works with any mask type
        template<typename MaskType>
        SIMD_FORCEINLINE int CountTrailingZeros(MaskType mask) noexcept
        {
            if (!mask) return sizeof(MaskType) * 8;
            
#if defined(SIMD_COMPILER_MSVC)
            unsigned long idx;
            if constexpr (sizeof(MaskType) <= 4)
            {
                _BitScanForward(&idx, static_cast<unsigned long>(mask));
            }
            else
            {
                #ifdef _WIN64
                _BitScanForward64(&idx, static_cast<unsigned long long>(mask));
                #else
                if (static_cast<uint32_t>(mask))
                {
                    _BitScanForward(&idx, static_cast<uint32_t>(mask));
                }
                else
                {
                    _BitScanForward(&idx, static_cast<uint32_t>(mask >> 32));
                    idx += 32;
                }
                #endif
            }
            return static_cast<int>(idx);
#elif SIMD_HAS_BUILTIN(__builtin_ctz) || SIMD_HAS_BUILTIN(__builtin_ctzll)
            if constexpr (sizeof(MaskType) <= 4)
            {
                return __builtin_ctz(static_cast<unsigned>(mask));
            }
            else
            {
                return __builtin_ctzll(static_cast<unsigned long long>(mask));
            }
#else
            int count = 0;
            while ((mask & 1) == 0)
            {
                mask >>= 1;
                ++count;
            }
            return count;
#endif
        }

        // Hardware CRC32 or high-quality fallback
        SIMD_FORCEINLINE uint64_t HashCombine(uint64_t seed, uint64_t value) noexcept
        {
#if defined(SIMD_HAS_SSE42)
            return _mm_crc32_u64(seed, value);
#elif defined(SIMD_HAS_ARM_CRC32)
            return __crc32cd(static_cast<uint32_t>(seed), value);
#else
            // MurmurHash3 finalizer
            uint64_t h = seed ^ value;
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return h;
#endif
        }

        // Prefetch operations
        // 
        // Optimal prefetch distance depends on:
        // - Memory bandwidth vs computation ratio
        // - Cache sizes (L1: ~32KB, L2: ~256KB, L3: ~8MB typical)
        // - Access pattern (sequential vs random)
        //
        // Guidelines:
        // - Light computation: prefetch 2-4 cache lines ahead (~128-256 bytes)
        // - Heavy computation: prefetch 8-16 cache lines ahead (~512-1024 bytes)
        // - Use T0 for data needed in next few iterations
        // - Use T1 for data needed in ~10-50 iterations
        // - Use T2 for data needed in ~100+ iterations
        // - Use NTA for streaming data that won't be reused
        //
        // Example distances for different scenarios:
        // - Simple byte matching: prefetch 256-512 bytes ahead
        // - Complex pattern matching: prefetch 1-2KB ahead
        // - Video/image processing: prefetch next row/tile
        //
        enum class PrefetchHint
        {
            T0 = 0,  // Prefetch to all cache levels (use for immediate data)
            T1 = 1,  // Prefetch to L2 and higher (use for near-future data)
            T2 = 2,  // Prefetch to L3 and higher (use for far-future data)
            NTA = 3  // Non-temporal (use for streaming, write-once data)
        };

        SIMD_FORCEINLINE void PrefetchRead(const void* ptr, PrefetchHint hint = PrefetchHint::T0) noexcept
        {
#if defined(SIMD_ARCH_X64) || defined(SIMD_ARCH_X86)
#if defined(SIMD_COMPILER_MSVC)
            switch (hint)
            {
                case PrefetchHint::T0:  _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0); break;
                case PrefetchHint::T1:  _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T1); break;
                case PrefetchHint::T2:  _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T2); break;
                case PrefetchHint::NTA: _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_NTA); break;
            }
#else
            switch (hint)
            {
                case PrefetchHint::T0:  __builtin_prefetch(ptr, 0, 3); break;
                case PrefetchHint::T1:  __builtin_prefetch(ptr, 0, 2); break;
                case PrefetchHint::T2:  __builtin_prefetch(ptr, 0, 1); break;
                case PrefetchHint::NTA: __builtin_prefetch(ptr, 0, 0); break;
            }
#endif
#elif SIMD_HAS_BUILTIN(__builtin_prefetch)
            int locality = 3 - static_cast<int>(hint);
            __builtin_prefetch(ptr, 0, locality);
#elif defined(SIMD_HAS_NEON) && defined(__ARM_FEATURE_UNALIGNED)
            __pld(ptr);
#else
            (void)ptr;
            (void)hint;
#endif
        }

        SIMD_FORCEINLINE void PrefetchT0(const void* ptr) noexcept { PrefetchRead(ptr, PrefetchHint::T0); }
        SIMD_FORCEINLINE void PrefetchT1(const void* ptr) noexcept { PrefetchRead(ptr, PrefetchHint::T1); }
        SIMD_FORCEINLINE void PrefetchT2(const void* ptr) noexcept { PrefetchRead(ptr, PrefetchHint::T2); }
        SIMD_FORCEINLINE void PrefetchNTA(const void* ptr) noexcept { PrefetchRead(ptr, PrefetchHint::NTA); }

        // ============== Batch Operations ==============
        
        // Process multiple vectors in a batch for better performance
        template<SimdWidth Width, size_t BatchSize = 4>
        struct BatchOps
        {
            using MaskType = typename WidthTraits<Width>::MaskType;
            static constexpr size_t stride = WidthTraits<Width>::bytes;
            
            // Batch match with prefetching
            static SIMD_FORCEINLINE void MatchByteMaskBatch(
                const void* data,
                uint8_t value,
                MaskType* results,
                size_t count) noexcept
            {
                const uint8_t* ptr = static_cast<const uint8_t*>(data);
                
                // Process in batches with prefetching
                size_t i = 0;
                for (; i + BatchSize <= count; i += BatchSize)
                {
                    // Prefetch next batch
                    if (i + BatchSize < count)
                    {
                        PrefetchT0(ptr + (i + BatchSize) * stride);
                        if constexpr (BatchSize >= 2)
                            PrefetchT0(ptr + (i + BatchSize + 1) * stride);
                    }
                    
                    // Process current batch - unrolled
                    results[i] = MatchByteMask<Width>(ptr + i * stride, value);
                    if constexpr (BatchSize >= 2)
                        results[i + 1] = MatchByteMask<Width>(ptr + (i + 1) * stride, value);
                    if constexpr (BatchSize >= 4)
                    {
                        results[i + 2] = MatchByteMask<Width>(ptr + (i + 2) * stride, value);
                        results[i + 3] = MatchByteMask<Width>(ptr + (i + 3) * stride, value);
                    }
                }
                
                // Process remaining
                for (; i < count; ++i)
                {
                    results[i] = MatchByteMask<Width>(ptr + i * stride, value);
                }
            }
            
            // Batch match either with prefetching
            static SIMD_FORCEINLINE void MatchEitherByteMaskBatch(
                const void* data,
                uint8_t val1,
                uint8_t val2,
                MaskType* results,
                size_t count) noexcept
            {
                const uint8_t* ptr = static_cast<const uint8_t*>(data);
                
                size_t i = 0;
                for (; i + BatchSize <= count; i += BatchSize)
                {
                    // Prefetch next batch
                    if (i + BatchSize < count)
                    {
                        PrefetchT0(ptr + (i + BatchSize) * stride);
                        if constexpr (BatchSize >= 2)
                            PrefetchT0(ptr + (i + BatchSize + 1) * stride);
                    }
                    
                    // Process current batch
                    results[i] = MatchEitherByteMask<Width>(ptr + i * stride, val1, val2);
                    if constexpr (BatchSize >= 2)
                        results[i + 1] = MatchEitherByteMask<Width>(ptr + (i + 1) * stride, val1, val2);
                    if constexpr (BatchSize >= 4)
                    {
                        results[i + 2] = MatchEitherByteMask<Width>(ptr + (i + 2) * stride, val1, val2);
                        results[i + 3] = MatchEitherByteMask<Width>(ptr + (i + 3) * stride, val1, val2);
                    }
                }
                
                // Process remaining
                for (; i < count; ++i)
                {
                    results[i] = MatchEitherByteMask<Width>(ptr + i * stride, val1, val2);
                }
            }
            
            // Find first match in batch
            static SIMD_FORCEINLINE int FindFirstMatchInBatch(
                const void* data,
                uint8_t value,
                size_t count) noexcept
            {
                const uint8_t* ptr = static_cast<const uint8_t*>(data);
                
                for (size_t i = 0; i < count; ++i)
                {
                    auto mask = MatchByteMask<Width>(ptr + i * stride, value);
                    if (mask)
                    {
                        int bit_pos = CountTrailingZeros(mask);
                        return static_cast<int>(i * stride + bit_pos);
                    }
                    
                    // Prefetch ahead
                    if (i + 2 < count)
                        PrefetchT1(ptr + (i + 2) * stride);
                }
                
                return -1; // Not found
            }
        };
    }
} // namespace Simd