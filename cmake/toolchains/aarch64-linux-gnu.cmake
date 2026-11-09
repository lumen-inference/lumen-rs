# Debian/Ubuntu cross tools: gcc-aarch64-linux-gnu and qemu-user.
# Override the cache variables for another installation prefix.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(VELRI_QEMU_AARCH64 qemu-aarch64 CACHE FILEPATH "AArch64 user-mode emulator")
set(VELRI_ARM_RUNTIME /usr/aarch64-linux-gnu CACHE PATH "AArch64 runtime for QEMU")
set(CMAKE_CROSSCOMPILING_EMULATOR "${VELRI_QEMU_AARCH64};-L;${VELRI_ARM_RUNTIME}")
