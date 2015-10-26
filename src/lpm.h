#ifndef _IXMAPFWD_LPM_H
#define _IXMAPFWD_LPM_H

#include "linux/list.h"

#define TABLE_SIZE_16 (1 << 16)
#define TABLE_SIZE_8 (1 << 8)

struct lpm_entry {
	struct list_head	list;
	void			*ptr;
	struct ixmap_marea	*area;
};

struct lpm_node {
	struct list_head	head;
	struct lpm_node		*next_table;
	struct ixmap_marea	*area;
};

struct lpm_table {
	struct lpm_node		node[TABLE_SIZE_16];
	void			(*entry_dump)(
				struct list_head *
				);
	int			(*entry_identify)(
				void *,
				unsigned int,
				unsigned int
				);
	int			(*entry_compare)(
				void *,
				unsigned int
				);
	void	 		(*entry_pull)(
				void *
				);
	void			(*entry_put)(
				void *
				);
};

#ifdef __CUDACC__
__device__ struct lpm_entry *lpm_lookup(struct lpm_table *table,
	void *dst);
#endif

void lpm_init(struct lpm_table *table);
int lpm_add(struct lpm_table *table, void *prefix,
	unsigned int prefix_len, unsigned int id,
	void *ptr, struct ixmap_desc *desc);
int lpm_delete(struct lpm_table *table, void *prefix,
	unsigned int prefix_len, unsigned int id);
void lpm_delete_all(struct lpm_table *table);
int lpm_traverse(struct lpm_table *table, void *prefix,
	unsigned int prefix_len);

#endif /* _IXMAPFWD_LPM_H */
