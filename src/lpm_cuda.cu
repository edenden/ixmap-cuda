#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <stddef.h>

extern "C" {
#include "linux/list_cuda.h"
#include "main.h"
#include "lpm.h"
#include "misc.h"
}

extern "C"
__device__ static unsigned int lpm_index_cuda(void *prefix,
	unsigned int offset, unsigned int range);

extern "C"
__device__ static unsigned int lpm_index_cuda(void *prefix,
	unsigned int offset, unsigned int range)
{
	unsigned int shift, word, mask;

	shift = 32 - ((offset % 32) + range);
	word = offset >> 5;
	mask = (1 << range) - 1;

	return (bswap_32(((uint32_t *)prefix)[word]) >> shift) & mask;
}

extern "C"
__device__ struct lpm_entry *lpm_lookup(struct lpm_table *table,
	void *dst)
{
	unsigned int index;
	struct lpm_node *node;
	struct list_head *head;
	struct lpm_entry *entry;
	unsigned int offset = 0;

	index = lpm_index_cuda(dst, 0, 16);
	node = &table->node[index];
	head = &node->head;
	offset += 16;

	while(node->next_table){
		index = lpm_index_cuda(dst, offset, 8);
		node = &node->next_table[index];

		if(!list_empty_cuda(&node->head)){
			head = &node->head;
		}
		offset += 8;
	}

	entry = list_first_entry_or_null(head, struct lpm_entry, list);
	return entry;
}

