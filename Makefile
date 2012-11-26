###############################################################################
#
# CUDA-SURF v0.5
# Copyright 2010 FUDAN UNIVERSITY
# Author: Max Lv
# Email: max.c.lv#gmail.com
#
################################################################################

# Add source files here
EXECUTABLE	   := refine
# CUDA source files (compiled with cudacc)
CUFILES_sm_11  := \
	match_kernel.cu

CCFILES        := \
	main.cpp \

COMMONFLAGS  += `pkg-config --cflags opencv`

#NVCCFLAGS += -ptx
LINKFLAGS += `pkg-config --libs opencv`

#verbose = 1

ROOTDIR := common

################################################################################
include common/common.mk
