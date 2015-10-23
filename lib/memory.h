#ifndef _IXMAP_MEMORY_H
#define _IXMAP_MEMORY_H

struct ixmap_marea {
	void			*ptr;
};

struct ixmap_mnode {
	struct ixmap_mnode	*child[2];
	struct ixmap_mnode	*parent;
	unsigned int		allocated;
	unsigned int		index;
	unsigned int		size;
	struct ixmap_marea	area;
};

struct ixmap_mnode *ixmap_mem_init(void *ptr, unsigned int size);
void ixmap_mem_destroy(struct ixmap_mnode *node);

#endif /* _IXMAP_MEMORY_H */
