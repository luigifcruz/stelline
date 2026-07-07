/*
 * Meson overlay substitute for the CMake-generated daqiri/version.h.
 * Values track the wrap revision (v2026.07.00). Update this file when
 * bumping the revision in subprojects/daqiri.wrap.
 */

#pragma once

#define DAQIRI_VERSION "2026.7.0"
#define DAQIRI_VERSION_YEAR 2026
#define DAQIRI_VERSION_MONTH 7
#define DAQIRI_VERSION_PATCH 0
#define DAQIRI_ABI_VERSION 0

namespace daqiri {

inline constexpr const char* version_string() noexcept { return DAQIRI_VERSION; }
inline constexpr int version_year() noexcept { return DAQIRI_VERSION_YEAR; }
inline constexpr int version_month() noexcept { return DAQIRI_VERSION_MONTH; }
inline constexpr int version_patch() noexcept { return DAQIRI_VERSION_PATCH; }
inline constexpr int abi_version() noexcept { return DAQIRI_ABI_VERSION; }

}  // namespace daqiri
