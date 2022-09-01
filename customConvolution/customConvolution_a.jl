#based on https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
# and https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
#so basically convolution is matrix multiplication of kernel and chunk of the array around all pixels that is the same size as the kernel
#about channels ...
# """ 
# Checking if there are mutliple channels for the single filter.
# If so, then each channel will convolve the image.
# The result of all convolutions are summed to return a single feature map.
# """