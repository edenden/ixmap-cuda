#ifndef _IXMAP_CUDA_H
#define _IXMAP_CUDA_H

struct ixmap_desc *ixmap_desc_alloc_cuda(struct ixmap_handle **ih_list,
	int ih_num, int queue_index);
void ixmap_desc_release_cuda(struct ixmap_handle **ih_list, int ih_num,
	int queue_index, struct ixmap_desc *desc);
struct ixmap_buf *ixmap_buf_alloc_cuda(struct ixmap_handle **ih_list,
	int ih_num, uint32_t count, uint32_t buf_size);
void ixmap_buf_release_cuda(struct ixmap_buf *buf,
	struct ixmap_handle **ih_list, int ih_num);

#endif /* _IXMAP_CUDA_H */
