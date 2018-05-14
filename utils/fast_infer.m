%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Updates the feature maps for a single training sample (image) without using
% the IPP libraries (and thus is SLOW). This is done via
% conjuagte gradient. This solves Ax=b where A is the F k matrix, x is the z feature maps
% and b is the y_tilda reconstruction (which you want to make equal y).
% Therefore Atb is F'y and AtAx is F'Fz (where each is a convolution).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @inference_file @copybrief fast_infer.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief fast_infer.m
%
% @param max_it number of conjugate gradient iterations
% @param z the feature maps to update (xdim+filter_size x ydim+filter_size x num_feature_maps).
% @param w the auxilary variable (same size as z).
% @param y the input maps for the layer (xdim x ydim x num_input_maps).
% @param F the filters (Fxdim x Fydim x num_input_maps x num_feature_maps).
% @param z0 the z0 feature maps (may not be used) (xdim+filter_size x
% ydim+filter_size x num_input_maps).
% @param z0_filter_size the size of the z0 filters (if used).
% @param lambda the coefficient on the reconstruction error term.
% @param beta the continuation variable on the ||z-x|| term.
% @param C the connectivity matrix for the layer.
% @param TRAIN_Z0 binary indicating if z0 should be used or not.
%
% @retval z the updated feature maps.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z] = fast_infer(max_it,z,w,y,F,z0,z0_filter_size,lambda,beta,C,TRAIN_Z0)

% Get the number of ks.
num_feature_maps = size(F,4);
num_input_maps = size(F,3);
xdim = size(y,1);
ydim = size(y,2);

% Initialize the running sum for each feature map.
conctemp = zeros(size(z));

%%%%%%%%%%
%%Compute the right hand side (A'b) term
% Do the f'y convolutions.
for j=1:num_input_maps
    if(TRAIN_Z0) % Convolve z0 map for j with it's filter.
        z0conv = conv2(z0(:,:,j),ones(z0_filter_size,z0_filter_size)/z0_filter_size,'valid');
    end
    for k = 1:num_feature_maps
        if(C(j,k)==1)           
            if(TRAIN_Z0) % Also convolve flipped Fjk with z0 maps convolution
                conctemp(:,:,k) = conctemp(:,:,k) +...
                    conv2(y(:,:,j),flipud(fliplr(F(:,:,j,k))),'full') -...
                    conv2(z0conv,flipud(fliplr(F(:,:,j,k))),'full');
            else
                % Place in correct location so when conctemp(:) is used below it will be
                % the correct vectorized form for dfz.
                conctemp(:,:,k) = conctemp(:,:,k) +...
                    conv2(y(:,:,j),flipud(fliplr(F(:,:,j,k))),'full');
            end
        end
    end
end
% This is the RHS. Only comput this once.
Atb = lambda*conctemp(:) + beta*w(:);
%%%%%%%%%%



%%%%%%%%%%
%%Compute the left hand side (A'Ax) term
% Initialize the running sum for each feature map.
conctemp = zeros(size(z));

for j=1:num_input_maps
    % Initialize a variable to keep the running some of the other convolutions
    % between f*z.
    convsum = zeros(xdim,ydim);
    
    % Loop over all the other ks and compute the sume of their
    % convolutions (f*z). This is the Ax term.
    for k = 1:num_feature_maps
        if(C(j,k)==1)
            % Convolve F k with z feature map and comput runnign sum.
            convsum = convsum + conv2(z(:,:,k),F(:,:,j,k),'valid');
        end
    end

    % This is the A'Ax term.
    for k = 1:num_feature_maps
        if(C(j,k)==1)
            % Place in correct location so when conctemp(:) is used below it will be
            % the correct vectorized form for dfz.
            conctemp(:,:,k) = conctemp(:,:,k) +...
                conv2(convsum,flipud(fliplr(F(:,:,j,k))),'full');
        end
    end
end
% This is the left hand side.
AtAx = lambda*conctemp(:)+beta*z(:);
%%%%%%%%%%


% Compute the residual.
r = Atb - AtAx;

for iter = 1:max_it
    rho = (r(:)'*r(:));
    
    if ( iter > 1 ),                       % direction vector
        %         their_beta = rho / rho_1;
        their_beta = double(abs(rho_1) > 1e-9).*rho / rho_1;  % Added from dilips.m
        p(:) = r(:) + their_beta*p(:);
    else
        p = r;
        p = reshape(p,size(w));
    end
    
    %%%%%%%%%%
    %%Compute the left hand side (A'Ax) term
    % Initialize the running sum for each feature map.
    conctemp = zeros(size(z));
    for j=1:num_input_maps
        % Initialize a variable to keep the running some of the other convolutions
        % between f*z.
        convsum = zeros(xdim,ydim);
        
        % Loop over all the other ks and compute the sume of their
        % convolutions (f*z). This is the Ax term.
        for k = 1:num_feature_maps
            if(C(j,k)==1)
                % Convolve F k with z feature map and comput runnign sum.
                convsum = convsum + conv2(p(:,:,k),F(:,:,j,k),'valid');
            end
        end

        % This is the A'Ax term.
        for k = 1:num_feature_maps
            if(C(j,k)==1)
                % Place in correct location so when conctemp(:) is used below it will be
                % the correct vectorized form for dfz.
                conctemp(:,:,k) = conctemp(:,:,k) +...
                    conv2(convsum,flipud(fliplr(F(:,:,j,k))),'full');
            end
        end
    end
    % This is the left hand side.
    q = lambda*conctemp(:)+beta*p(:);
    %%%%%%%%%%
    
    %     their_alpha = rho / (p(:)'*q(:) );
    temp = p(:)'*q(:);
    their_alpha = double(abs(temp) > 1e-9).*rho / temp;
    z(:) = z(:) + their_alpha * p(:);           % update approximation vector
    r = r - their_alpha*q;                      % compute residual
    
    rho_1 = rho;
%                 fprintf('\nIteration %d |residual| %.3g', iter, norm(r));
    if(norm(r) < 1e-6)
        %        fprintf('Convergence reached in iteration %d - breaking\n', iter);
        break;
    end;
end

end