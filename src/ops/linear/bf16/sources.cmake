target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n256_k5120.cu"
)

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_cublas.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/flash_next/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/flash_next/bf16_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/flash_next/bf16_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/flash_next/bf16_gemm_mma.cu")
target_link_libraries(ninfer_ops PRIVATE CUDA::cublas)
