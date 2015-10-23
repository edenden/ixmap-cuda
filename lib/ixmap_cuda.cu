__device__ uint8_t *ixmap_macaddr(struct ixmap_plane *plane,
	unsigned int port_index)
{
	return plane->ports[port_index].mac_addr;
}
