#ifdef __CUDACC__
extern "C"
__device__ uint16_t bswap_16(uint16_t x);
extern "C"
__device__ uint32_t bswap_32(uint32_t x);
extern "C"
__device__ int list_empty_cuda(const struct list_head *head);
#endif
