#ifndef _IXMAP_DRIVER_H
#define _IXMAP_DRIVER_H

/* TX descriptor defines */
#define IXGBE_MAX_TXD_PWR	14
#define IXGBE_MAX_DATA_PER_TXD	(1 << IXGBE_MAX_TXD_PWR)

/* Send Descriptor bit definitions */
#define IXGBE_TXD_STAT_DD	0x00000001 /* Descriptor Done */
#define IXGBE_TXD_CMD_EOP	0x01000000 /* End of Packet */
#define IXGBE_TXD_CMD_IFCS	0x02000000 /* Insert FCS (Ethernet CRC) */
#define IXGBE_TXD_CMD_RS	0x08000000 /* Report Status */
#define IXGBE_TXD_CMD_DEXT	0x20000000 /* Desc extension (0 = legacy) */

/* Adv Transmit Descriptor Config Masks */
#define IXGBE_ADVTXD_DTYP_DATA	0x00300000 /* Adv Data Descriptor */
#define IXGBE_ADVTXD_DCMD_IFCS	IXGBE_TXD_CMD_IFCS /* Insert FCS */
#define IXGBE_ADVTXD_DCMD_DEXT	IXGBE_TXD_CMD_DEXT /* Desc ext 1=Adv */
#define IXGBE_ADVTXD_PAYLEN_SHIFT \
				14 /* Adv desc PAYLEN shift */

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#define IXGBE_TX_DESC(R, i)	\
	(&(((union ixmap_adv_tx_desc *)((R)->addr_virt))[i]))

#endif /* _IXMAP_DRIVER_H */
