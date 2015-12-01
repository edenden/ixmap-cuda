#ifndef _IXMAP_CUDA_H
#define _IXMAP_CUDA_H

#ifdef __CUDACC__
extern "C"
__device__ uint8_t *ixmap_macaddr_cuda(struct ixmap_plane *plane,
	unsigned int port_index);
extern "C"
__host__ void ixmap_slot_release_cuda(struct ixmap_buf *buf,
	int slot_index);
#endif

struct ixmap_desc *ixmap_desc_alloc_cuda(struct ixmap_handle **ih_list,
	int ih_num, int queue_index);
void ixmap_desc_release_cuda(struct ixmap_handle **ih_list, int ih_num,
	int queue_index, struct ixmap_desc *desc);
struct ixmap_buf *ixmap_buf_alloc_cuda(struct ixmap_handle **ih_list,
	int ih_num, uint32_t count, uint32_t buf_size);
void ixmap_buf_release_cuda(struct ixmap_buf *buf,
	struct ixmap_handle **ih_list, int ih_num);

struct nvmap_handle;

struct nvmap_handle *nvmap_open(void);
void nvmap_close(struct nvmap_handle *nh);
struct ixmap_buf *ixmap_buf_alloc_gpu(struct nvmap_handle *nh,
	int ih_num, uint32_t count, uint32_t buf_size);
void ixmap_buf_release_gpu(struct ixmap_buf *buf,
	struct nvmap_handle *nh, int ih_num);

#endif /* _IXMAP_CUDA_H */
