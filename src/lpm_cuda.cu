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
#include "lpm_cuda.h"
}

__device__ uint32_t bswap_32(uint32_t x) {
	x = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0x00FF00FF);
	return (x >> 16) | (x << 16);
}

__device__ unsigned int lpm_index_cuda(void *prefix,
	unsigned int offset, unsigned int range)
{
	unsigned int shift, word, mask;

	shift = 32 - ((offset % 32) + range);
	word = offset >> 5;
	mask = (1 << range) - 1;

	return (bswap_32(((uint32_t *)prefix)[word]) >> shift) & mask;
}

__device__ int list_empty_cuda(const struct list_head *head)
{
	return head->next == head;
}

__global__ struct lpm_entry *lpm_lookup(struct lpm_table *table,
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

