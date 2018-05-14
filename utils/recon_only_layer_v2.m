%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This is a combined function to both train or reconstruct images
% in the pixel space. The type of operation is determined by model.exptype
% Reconstructs a single layer (specified by model.layer) of the model
% based on the input y. y is the input maps of model.layer which maybe be
% the feature maps of a layer below. The feature maps are inferred from
% this and then used to reconstruct a y' reconstruction of the input maps.
% Inputs: The inputs are passed in varargin (a cell array) because when
% reconstructing from higher layers, all the lower layers need to be passed
% in to reconstruct down to the lowest level.
% @copybrief train_recon_layer.m
%
% @param varargin should be of the following format: (example for 2 layers)
% \li \e model2 struct with parameters for top layer
% \li \e F2 third layer filters or []
% \li \e z02 third layer z0 maps or []
% \li \e pooled_inpdices2 the indices from Max pooling after L2 (usually not every used) to allow reconstructions
% \li \e model1 struct with parameters for layer 1
% \li \e F1 layer 1 filters
% \li \e z02 fist layer z0 maps or []
% \li \e pooled_inpdices1 the indices from Max pooling after L1 (usually notevery used) to allow reconstructions
% \li \e pooled_inpdices0 the indices from Max pooling on the image (usually not every used) to allow reconstructions
% \li \e y the input maps for the given layer (may be noisy if denoising)
% \li \e original_images the clean images (will be identical to y when training on clean images)
%
% @retval F the learned (or previous if reconstructing) filters.
% @retval z the inferred feature maps.
% @retval z0 the inferred z0 feature maps (or [] if not used).
% @retval recon_images the reconstructed images (required DISPLAY_ERROR to be
% set).
% @retval model some fields of the model structure will be updated within (with
% xdim, ydim, and errors for example).

function [F,z,z0,recon_images,model] = recon_only_layer_v2(varargin)

%% Read in the variables arguments
% The layer to be inferred is the first on input.
layer = varargin{1}.layer;
% Set up the top layer's variables.
model = varargin{1};
F = varargin{2};
% z0 = varargin{3};% Get the model,F,z0 triples
% The actual model that is being inferred is stored in model,F,z0.% All the layers are stored in model#,z#,z0#
for i=1:layer
    eval(strcat('model',num2str(layer-i+1),'=varargin{(i-1)*3+1};'));
    eval(strcat('F',num2str(layer-i+1),'=varargin{(i-1)*3+2};'));
    eval(strcat('z0',num2str(layer-i+1),'=varargin{(i-1)*3+3};'));    
end
z_init = varargin{end-2};
y = varargin{end-1};% Get the input maps for this layer. (for teh first layer this is the same as the noisy input image planes).
original_images = varargin{end};% Get the original image.
model.xdim = size(y,1); % Assuming square images.
eval(strcat('model',num2str(layer),'.xdim = size(y,1);'));
xdim = model.xdim;
model.ydim = size(y,2); % Assuming square images.
eval(strcat('model',num2str(layer),'.ydim = size(y,2);'));
ydim = model.ydim;
model.num_input_maps(layer) = size(y,3); % Set the number of input maps here.
%% Parameters. DO NOT CHANGE HERE. Change in the gui.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filter_size = model.filter_size(model.layer);% Size of each filter
num_feature_maps = model.num_feature_maps(model.layer);% Number of filters to learn
maxepochs = model.maxepochs(model.layer);% Number of epochs total through the training set.
min_iterations = model.min_iterations;% Number of iterations to run minimize
lambda = model.lambda(model.layer);% Theshold to control the gradients grad_threshold = model.grad_threshold;Sparsity Weighting
lambda_input = model.lambda_input;
% Use 0.01, batch, and put lambda = lambda + 0.1 to see that the first filters just take patches.
% RAMP_LAMBDA = model.RAMP_LAMBDA;
% ramp_lambda_amount = model.ramp_lambda_amount;
% Dummy regularization weighting. This starts at beta=Binitial, increases
% by beta*Bmultiplier each iteration until T iterations are done.
Binitial = model.Binitial;
Bmultiplier = model.Bmultiplier;
betaT = model.betaT;
beta = Binitial;
% % See if ramping the beta back down after half of T's iterations helps.% RAMP_DOWN_AND_UP = model.RAMP_DOWN_AND_UP;% The alpha-norm on the dummy regularizer variables
alpha = model.alpha(model.layer);
% The alpha-norm on the filters% alphaF = model.alphaF;% The coefficient on the sparsity term in for each layer.
kappa = model.kappa;
% Not implemented yet% The normalization on the ||z-x|| dummy variable term.% beta_norm = model.beta_norm;
% Connectivity Matrix, this is a input_maps by num_feature_maps matrix.
connectivity_matrix = model.conmats{model.layer};
TRAIN_Z0 = model.TRAIN_Z0;% If the  z0 map is used at this layer.
layer = model.layer;
y_input = y;
%% TRAINING PHASE SETUP
% if z_init == 0
    z =zeros((xdim+filter_size-1),(ydim+filter_size-1),num_feature_maps,'single');
% else
%     z = z_init;
% end

z0=0;% If using the z0 for the given layer.
z0_filter_size = 1;
clear varargin;% Get rid of the input parameters (to save memory)
w = z;% Introduce the dummy variables w, same size as z.
recon_images = zeros(size(original_images),'single');% Initialize a matrix to store the reconstructed images.
% if(layer==1)
%     f = figure(200+model.layer); clf;% Display the filters
%     sdispfilts(F,model.conmats{model.layer}); set(f,'Name',strcat('Layer ',num2str(model.layer),' Filters')); drawnow
% elseif(layer > 1) % Display the pixel space filters for higher layers.    
%     eval(strcat('F',num2str(layer),' = F;'));
%     eval(strcat('recon_z',num2str(layer),' = z;'));    
%     top_down_noload  % %%%% 这个函数的目的等于就是将你想计算的filter map 所对应的coeff的值为1，其余的为0Generate the pixel space filters.
%     f = figure(250+model.layer); clf; sdispims(recon_y1); set(f,'Name',strcat('Layer ',num2str(model.layer),' Pixel Filters'));  drawnow;
% end
for epoch = 1:maxepochs
    z0sample = 0; %fprintf('Layer: %1d, Epoch: %2d',model.layer,epoch);
    for beta_iteration = 1:betaT   %% Beta Regeme
        if beta_iteration == 1
            beta = Binitial;
        elseif beta_iteration == betaT
            beta = beta*Bmultiplier; %fprintf('.\n');     
        else
            beta = beta*Bmultiplier; %fprintf('.');       
        end
        w(:) = solve_image(z(:),beta,alpha,kappa(layer)); % %% Update the w values Generic code that does all the below.
        z = fast_infer(min_iterations,z,w,y,F,z0sample,z0_filter_size,lambda,beta,connectivity_matrix,TRAIN_Z0);%% Update Feature Maps
    end
    %% Compute Errors
    eval(strcat('F',num2str(layer),' = F;'));
    eval(strcat('recon_z',num2str(layer),' = z;'));
    top_down_core  %% Reconstruct from the top down.
    recon_images = recon_z0;
    %% Display reconstructed Image planes.
%     f = figure(500+model.layer); clf; sdispims(recon_images); set(f,'Name',strcat('Layer ',num2str(model.layer),' Reconstructions'));drawnow;
%     if epoch==1
%         cursize = get(f,'Position');  screensize = get(0,'Screensize'); set(f,'Position',[850+cursize(3)*model.layer,30,cursize(3),cursize(4)])
%     end
end
end



% %% Clear the error matrices (especially for reconstruction)
% model.update_noise_rec_error = [];
% model.pix_noise_rec_error = [];
% model.pix_clean_rec_error = [];
% model.pix_clean_SNR_error = [];
% model.pix_update_rec_error = [];
% model.reg_error = [];
% model.beta_rec_error = [];
% model.unscaled_total_energy = [];
% model.scaled_total_energy = [];
% 
%     %% Compute and Store Errors
%     % Compute error versus the input maps (recon_z# where # is the layer below.
%     eval(strcat('recon_error = sqrt(sum(sum(sum((recon_z',num2str(layer-1),'-y_input).^2))));'));
%     model.pix_noise_rec_error(epoch) = recon_error;
%     
%     % Layer 1's error versus the updated pixel space images.
%     eval(strcat('recon_error = sqrt(sum(sum(sum((recon_z',num2str(layer-1),'-y).^2))));'));
%     model.pix_update_rec_error(epoch) = recon_error;
%     upd_rec_error = sqrt(sum(sum(sum((y_input-y).^2))));
%     model.update_noise_rec_error(epoch) = upd_rec_error;
%     
%     % Compute regularization error.
%     reg_error = sum(abs(z(:))); % Compare with L1 norm.
%     model.reg_error(epoch) = reg_error;
%     
%     % Layer 1's error versus the clean pixel space images.
%     recon_error = sqrt(sum(sum(sum((recon_images-original_images).^2))));
%     SNR_error = compute_snr(original_images,recon_images);
%     fprintf(' ::: Pix Clean Total error: %4.1f, Recon error: %4.1f, Reg error: %4.1f SNR: %4.4f / Noisy: %4.4f = %3.1f x\n',...
%         recon_error+reg_error,recon_error,reg_error,SNR_error,noisy_SNR(1),SNR_error/noisy_SNR(1));
%     model.pix_clean_rec_error(epoch) = recon_error;
%     model.pix_clean_SNR_error(epoch) = SNR_error;
%     
%     % Compute Beta reconstruction error.
%     recon_error = sqrt(sum(sum(sum((w-z).^2))));
%     model.beta_rec_error(epoch) = recon_error;
%     
%     % Compute the sum of each term (without coefficients).
%     model.unscaled_total_energy(epoch) = model.pix_update_rec_error(epoch)+model.reg_error(epoch)+upd_rec_error+model.beta_rec_error(epoch);
%     fprintf(' ::: Energy Function Total: %4.1f Lay Reg: %4.1f Update Error: %4.1f Lay Rec Upd: %4.1f Lay Rec Beta: %4.1f\n',...
%         model.unscaled_total_energy(epoch),model.reg_error(epoch),upd_rec_error,model.pix_update_rec_error(epoch),model.beta_rec_error(epoch))
%     
%     % Compute the sum of each term (with coefficients)
%     model.scaled_total_energy(epoch) = lambda/2*model.pix_update_rec_error(epoch)+kappa(layer)*model.reg_error(epoch)+lambda_input/2*upd_rec_error+(beta/2)/kappa(layer)*model.beta_rec_error(epoch);
%     fprintf(' ::: Scaled Energy F Total: %4.1f Lay Reg: %4.1f Update Error: %4.1f Lay Rec Upd: %4.1f Lay Rec Beta: %4.1f\n',...
%         model.scaled_total_energy(epoch),kappa(layer)*model.reg_error(epoch),lambda_input*upd_rec_error,lambda/2*model.pix_update_rec_error(epoch),(beta/2)*model.beta_rec_error(epoch))
% 
