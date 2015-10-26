#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern "C" {
#include "linux/list_cuda.h"
}

__device__ uint16_t bswap_16(uint16_t x)
{
	x = (x>>8) | (x<<8);
	return x;
}

__device__ uint32_t bswap_32(uint32_t x)
{
	x = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0x00FF00FF);
	return (x >> 16) | (x << 16);
}

__device__ int list_empty_cuda(const struct list_head *head)
{
	return head->next == head;
}
