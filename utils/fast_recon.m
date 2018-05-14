%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Reconstructs the input maps from the feature maps convolved with the filters 
% (and possibly z0 maps as well) (image) without using
% the IPP libraries (and thus is SLOW).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @recon_file @copybrief fast_recon.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief fast_recon.m
%
% @param z0 the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps).
% @param z0_filter_size the size of the z0 filters (if used).
% @param z the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps).
% @param F the filters (Fxdim x Fydim x num_input_maps x num_feature_maps).
% @param C the connectivity matrix for the layer.
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
% @param COMP_THREADS the number of threads to split computation over.
%
% @retval I the reconstructed input maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I] = fast_recon(z0,z0_filter_size,z,F,C,TRAIN_Z0)
% Note this assumes that not thresholding of the variance has been done.

num_feature_maps = size(F,4);
num_input_maps = size(F,3);
filter_size = size(F,1);
xdim = size(z,1)-filter_size+1;
ydim = size(z,2)-filter_size+1;
I = zeros(xdim,ydim,num_input_maps);

for j=1:num_input_maps
    % Initialize a variable to keep the running some of the other convolutions
    % between f*z.
    convsum = zeros(xdim,ydim);
    
    % Loop over all the other filters and compute the sume of their
    % convolutions (f*z).
    for k = 1:num_feature_maps
        if(C(j,k) == 1) % Only do convolutions where connected.
            
            % Convolve F(:,other) filter with z(:,other) feature map.
            convsum = convsum + conv2(z(:,:,k),F(:,:,j,k),'valid');
        end
    end
    if(TRAIN_Z0)
            convsum = convsum + conv2(z0(:,:,j),ones(z0_filter_size,z0_filter_size)/z0_filter_size,'valid'); % This is column normalized.
    end
    
    I(:,:,j) = convsum(:,:);
    
    
end

end
