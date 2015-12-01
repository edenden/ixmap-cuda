NVIDIA_VERSION	:= $(shell modinfo nvidia | grep version: | awk '{printf("%s", $$2)}')
KBUILD_EXTRA_SYMBOLS := /usr/src/nvidia-$(NVIDIA_VERSION)/Module.symvers
EXTRA_CFLAGS := -Wall -Wunused-function -I/usr/src/nvidia-$(NVIDIA_VERSION)/

# obj-m is a list of what kernel modules to build.  The .o and other
# objects will be automatically built from the corresponding .c file -
# no need to list the source files explicitly.
obj-m := nvmap.o
nvmap-objs := nvmap_main.o nvmap_fops.o nvmap_dma.o

# KDIR is the location of the kernel source.  The current standard is
# to link to the associated source tree from the directory containing
# the compiled modules.
KDIR  := /lib/modules/$(shell uname -r)/build

# PWD is the current working directory and the location of our module
# source files.
PWD   := $(shell pwd)

# default is the default make target.  The rule here says to run make
# with a working directory of the directory containing the kernel
# source and compile only the modules in the PWD (local) directory.
default:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
