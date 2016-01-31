#ifndef _IXMAP_CUDA_H
#define _IXMAP_CUDA_H

#ifdef __CUDACC__
extern "C"
__device__ uint8_t *ixmap_macaddr_cuda(struct ixmap_plane_cuda *plane,
	unsigned int port_index);
#endif

struct ixmap_buf *ixmap_buf_alloc_cuda(struct ixmap_handle **ih_list,
	int ih_num, uint32_t count, uint32_t buf_size);
void ixmap_buf_release_cuda(struct ixmap_buf *buf,
	struct ixmap_handle **ih_list, int ih_num);

struct nvmap_handle;

struct nvmap_handle *nvmap_open(void);
void nvmap_close(struct nvmap_handle *nh);
struct ixmap_buf *ixmap_buf_alloc_cuda_direct(struct nvmap_handle *nh,
	int ih_num, uint32_t count, uint32_t buf_size);
void ixmap_buf_release_cuda_direct(struct ixmap_buf *buf,
	struct nvmap_handle *nh, int ih_num);

struct ixmap_plane_cuda *ixmap_plane_alloc_cuda(struct ixmap_handle **ih_list,
        struct ixmap_buf *buf, int ih_num, int queue_index);
void ixmap_plane_release_cuda(struct ixmap_plane_cuda *plane);

#endif /* _IXMAP_CUDA_H */
