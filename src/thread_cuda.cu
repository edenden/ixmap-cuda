#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <net/ethernet.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <pthread.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <stddef.h>
#include <syslog.h>
#include <ixmap.h>

extern "C" {
#include "main.h"
#include "thread.h"
#include "forward.h"
#include "epoll.h"
#include "netlink.h"
}

int thread_wait(struct ixmapfwd_thread *thread,
	int fd_ep, uint8_t *read_buf, int read_size)
{
        struct epoll_desc *ep_desc;
        struct ixmap_irqdev_handle *irqh;
        struct epoll_event events[EPOLL_MAXEVENTS];
	struct ixmap_packet packet[IXMAP_RX_BUDGET];
        int i, ret, num_fd;
        unsigned int port_index;

	while(1){
		num_fd = epoll_wait(fd_ep, events, EPOLL_MAXEVENTS, -1);
		if(num_fd < 0){
			goto err_read;
		}

		for(i = 0; i < num_fd; i++){
			ep_desc = (struct epoll_desc *)events[i].data.ptr;
			
			switch(ep_desc->type){
			case EPOLL_IRQ_RX:
				irqh = (struct ixmap_irqdev_handle *)ep_desc->data;
				port_index = ixmap_port_index(irqh);

				/* Rx descripter cleaning */
				ret = ixmap_rx_clean(thread->plane, port_index,
					thread->buf, packet);

				forward_process<<<1, 512>>>(thread, port_index, packet);

				for(i = 0; i < thread->num_ports; i++){
					ixmap_tx_xmit(thread->plane, i);
				}

				ret = read(ep_desc->fd, read_buf, read_size);
				if(ret < 0)
					goto err_read;

				ixmap_irq_unmask_queues(thread->plane, irqh);
				break;
			case EPOLL_IRQ_TX:
				irqh = (struct ixmap_irqdev_handle *)ep_desc->data;
				port_index = ixmap_port_index(irqh);

				/* Tx descripter cleaning */
				ixmap_tx_clean(thread->plane, port_index, thread->buf);
				for(i = 0; i < thread->num_ports; i++){
					ixmap_rx_assign(thread->plane, i, thread->buf);
				}

				ret = read(ep_desc->fd, read_buf, read_size);
				if(ret < 0)
					goto err_read;

				ixmap_irq_unmask_queues(thread->plane, irqh);
				break;
			case EPOLL_TUN:
				port_index = *(unsigned int *)ep_desc->data;

				ret = read(ep_desc->fd, read_buf, read_size);
				if(ret < 0)
					goto err_read;

				forward_process_tun(thread, port_index, read_buf, ret);
				for(i = 0; i < thread->num_ports; i++){
					ixmap_tx_xmit(thread->plane, i);
				}
				break;
			case EPOLL_NETLINK:
				ret = read(ep_desc->fd, read_buf, read_size);
				if(ret < 0)
					goto err_read;

				netlink_process(thread, read_buf, ret);
                                break;
			case EPOLL_SIGNAL:
				ret = read(ep_desc->fd, read_buf, read_size);
				if(ret < 0)
					goto err_read;

				goto out;
				break;
			default:
				break;
			}
		}
	}

out:
	return 0;

err_read:
	return -1;
}

