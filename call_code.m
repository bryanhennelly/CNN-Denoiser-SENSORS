clear all;
load('Network.mat');
%denoise test samples
clear data; 
%fileneame should contain the data to be denoised - size 600 x N where N is
%the number od spectra. Each spectrum should be a separate line in the text
%file, which values separated by commas. A sample file is provided.
%Alternatively you can simply provide your data as a matrix of size [600, M]
data=dlmread(['rawData.txt'],','); 
m=max(data,[],'all');
%testing samples are scaled in order to be consistent with the magnitue of
%the training data
scaling_factor=5000/m;
data=data'*scaling_factor; 
SIZE=size(data);
noisy_test_samples=reshape(data,[600 1 1 SIZE(2)]);
denoised=predict(dlnet,gpuArray(dlarray(noisy_test_samples,'SSCB'))); 
denoised_raw_data=gather(extractdata(denoised))/scaling_factor;
