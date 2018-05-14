%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Normalizes each plane of the input (assumed to be in the first two dimensions
% of a 3 or 4-D matrix). This makes the each plane have unit length.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @other_comp_file @copybrief plane_normalize.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief plane_normalize.m
%
% @param in the 3 or 4-D matrix in which the first two dimensions define a
% plane.
% @retval in the normalized version of the input matrix.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [in] = plane_normalize(in)

% The input should be planes by A by B (4-D) matrix. 
% The output is a normalized version of the input.
% The normalization is over each plane. 
% 
% Get original image sizes.
xdim = size(in,1);
ydim = size(in,2);

% Columnize the planes.
in = reshape(in,xdim*ydim,size(in,3),size(in,4));

% Normalize by summing over the squared columns (each plane).
% in = in ./ repmat(sum(in, 1),size(in,1),1);
in = in ./ repmat(sqrt(sum(in.^2, 1)),size(in,1),1);

% Reshape to the original size.
in = reshape(in,xdim,ydim,size(in,2),size(in,3));


% 
% % Normalize over all the connections it has to input maps.
% 
% % Get original image sizes.
% [xdim ydim numinp numfeat] = size(in);
% 
% % Columnize the planes.
% in = reshape(in,xdim*ydim*numinp,numfeat);
% 
% % Normalize by summing over the squared columns (each plane).
% % in = in ./ repmat(sum(in, 1),size(in,1),1);
% in = in ./ repmat(sqrt(sum(in.^2, 1)),size(in,1),1);
% 
% % Reshape to the original size.
% in = reshape(in,xdim,ydim,numinp,numfeat);




% 
% % Normalize over all the filters at once.
%%% This really destroys the filters (only one is used).
% in(:) = in(:) ./ sqrt(sum(in(:).^2, 1));
% 

end