%% The Code is created based on the method described in the following papers: 
% Q. Liu, H. Leung, "Synthesis-analysis deconvolutional network for compressed sensing," Image Processing (ICIP), 2017 IEEE International Conference on. IEEE, 2017: 1940-1944.
% Author: Q. Liu, H. Leung
% Date : 09/2018 
% Version : 1.0 
% The code and the algorithm are for non-comercial use only. 
% Copyright 2018, Department of Electronic Information Engineering, Nanchang University. 
% SADN - Synthesis-analysis deconvolutional network
% 
% Paras: 
%       1. alpha : sparse norm
%       2. Bmultiplier : continuation parameter
%       3. Binitial : continuation parameter
%       4. lambda : regularization  parameter
%       5. iter : Number of iterations
% Example 

clear; clc;

fprintf('Adding paths now...\n')
getd = @(p)path(path,p);% Add some directories to the path
getd('.\utils\');
getd('./train_AS0001_1854_alpha08_q1/');

% %% Set Model Parameters/Defaults
model.fullmodelpath = 'epoch5_layer2.mat';
fullmodelpath = model.fullmodelpath;
% Remove the '.mat' from the top model's path.
topmodelpath = remove_dot_mat(fullmodelpath);

% Checks how many epochs are in the fullmodelpath (after .mat is removed)
% The startpath is used as prefixes for the epochs that are read from the
% top_model.
if(strcmp(topmodelpath(end-8:end-8),'h')) %Single digit epochs
    startpath = topmodelpath(1:end-8);
elseif(strcmp(topmodelpath(end-9:end-9),'h')) % Double digit epochs
    startpath = topmodelpath(1:end-9);
else % Triple digit epochs
    startpath = topmodelpath(1:end-10);
end

% Load the top layer of the model.
% load(fullmodelpath,'model','F','z0');  % Only need these varaibles.
load(fullmodelpath);  % Only need these varaibles.

% Get where the pooling files are stored.
pooldir = parentdir(fullmodelpath);

% Save the top model and layer.
top_model = backwards_compatible(model);
top_layer = model.layer;
% Save the size of the top layer (for one training sample)
top_size = size(z0(:,:,:,1));


% Save the epochs of the layers below.
maxepochs = model.maxepochs;
% Save the variables of the top layer.
eval(strcat('model',num2str(model.layer),'=model;'));
eval(strcat('F',num2str(model.layer),'=single(F);'));
% eval(strcat('z',num2str(model.layer),'=z;'));
eval(strcat('z0',num2str(model.layer),'=single(z0);'));

% Save the variables for each of the other layers.
for layer=model.layer-1:-1:1
    % Load the model's below at the epochs for the top layer model.
    epoch = maxepochs(layer);
    try % Try if this file exists.
        load(strcat(startpath,num2str(epoch),'_layer',num2str(layer)));
    catch % Load the highest one you can find.
        load(strcat(startpath,num2str(get_highest_epoch(parentdir(startpath),layer)),'_layer',num2str(layer)));
    end
    % Make sure loaded layers are compatible.
    %     model = backwards_compatible(model);
    % Make new variable names based on the layer you are currently on.
    eval(strcat('model',num2str(model.layer),'=model;'));
    eval(strcat('F',num2str(model.layer),'=single(F);'));
    %         eval(strcat('z',num2str(model.layer),'=z;'));
    eval(strcat('z0',num2str(model.layer),'=single(z0);'));
end
clear z z0 F model maxepochs startpath epoch layer pooled_indices pooled_maps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F1 = F1(:,:,1,:);
model1.num_input_maps(1) = 1;model2.num_input_maps(1) = 1;
model1.conmats{1,1} = ones(1,18);model2.conmats{1,1} = ones(1,18);
%model1.lambda = [1826,1860,1,1];model2.lambda = model1.lambda;  %
model1.lambda = [2226,1860,1,1];model2.lambda = model1.lambda;  %
% model1.lambda = [1626,1860,1,1];model2.lambda = model1.lambda;  %
% model1.lambda = [1626,1860,1,1];model2.lambda = model1.lambda;  %
%model1.lambda = [1126,1860,1,1];model2.lambda = model1.lambda;  %31.72
model1.Binitial = 0.05; model1.Bmultiplier = 8;model1.betaT = 10;
model1.alpha(1) = 0.8;
model1.maxepochs = [1,1,2,2];model2.maxepochs = model1.maxepochs;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%% step1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M0 = imread('t2axialbrain.jpg');
M0 = im2double(M0);
M0 = imresize(M0,0.5);
if (length(size(M0))>2);  M0 = rgb2gray(M0);   end
figure(456); imshow(M0,[]);
[min(M0(:)),max(M0(:))]

% load mask 
n =size(M0,1);
load mask_all_random256.mat;
mask = fftshift(mask_all{1,3});
k = sum(sum(mask));
fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, k,1-k/n/n);
figure(455); imshow(mask,[]);

%#######%%%%% generate K-space data %%%%
sigma = 0;
y_noisy = mask.*(fft2(M0) + randn(n)*sigma*255 + (0+1i)*randn(n)*sigma*255); 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%% step2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 300; %
Im = abs(ifft2(y_noisy));   % 
figure(33);imagesc(Im);colormap(gray);axis off; axis equal;
f = y_noisy;
psnr_idex = zeros(1,iter);
npd = 1;fltlmbd = 1;
[sl_original, sh_original] = lowpass(M0, fltlmbd, npd);

sizeF = size(Im);
%bottom = 1;
filter_mu2 = 0.00001;
for iii = 1:size(F1,4)
    mcell.f{iii} = F1(:,:,1,iii);%         mcell.f_tr{iii} = flipud(fliplr(mcell.f{iii}));
    filter_eigsK(:,:,iii) = psf2otf(mcell.f{iii},sizeF);   %bottom = bottom + filter_mu2 * abs(filter_eigsK(:,:,iii)).^2;
end
M_mask = mask; M_mask(1,1)  = 1;  beta = 0.01;
%%%%%%%%%%%%%%%%%%%  main iteration  %%%%%%%%%%%%%%%%%%%%%
for iteri = 1:iter  %
    npd = 1;fltlmbd = 1;
    [sl, sh] = lowpass(Im, fltlmbd, npd);
    CSC_b = sh;
    
    %% Infer up for each layer (store reconstruction in y_tilda##) For each layer, have to infer z# fist (the y for the next layer).
    for recon_layer=1:1  %top_layer
        % Save the size of the original images (for undoing pooling)
        eval(strcat('model',num2str(recon_layer),'.orig_xdim = size(sh_original,1);'));
        eval(strcat('model',num2str(recon_layer),'.orig_ydim = size(sh_original,2);'));
        %%%%%%%    %% Make sure each model is backwards compatible.    %%%%%%%
        modelargs = ''; % A string of parameters to pass to each layer.
        for layer=recon_layer:-1:1% Construct the modelargs string.
            modelargs = strcat(modelargs,',','model',num2str(layer));
            modelargs = strcat(modelargs,',','F',num2str(layer));
            modelargs = strcat(modelargs,',','z0',num2str(layer));
        end
        modelargs = modelargs(2:end);    % Get rid of the first ',' that is in the string.
        modelargs = strcat(modelargs,',y,original_images');    % Add the input_map (y), original image and the noisy image.
        %fprintf('Reconstructing Layer %d of a %d-Layer Model\n',recon_layer,top_layer);
        
        %%%% The z0# for the layer is saved. y are the returned (inferred) feature maps (input maps for next layer)
        %     eval(strcat('[blahF,y,z0',num2str(recon_layer),',y_tilda',num2str(recon_layer),',model',num2str(recon_layer),'] = train_recon_layer(',modelargs,');'))
        if recon_layer == 1
            [blahF,CSC_b,z01,y_tilda1,model1] = recon_only_layer_v2(model1,F1,z01,CSC_b,sh_original);
        elseif recon_layer == 2
            [blahF,CSC_b,z02,y_tilda2,model2] = recon_only_Twolayer_v2(model2,F2,z02,model1,F1,z01,CSC_b,sh);
        end
        modelargs = '';   % Reset the modelargs sring.
    end
    %         figure(66);imshow([sh,y_tilda1],[])  %,y_tilda2
    temp_Im1 = y_tilda1 + sl;      %temp_Im = y_tilda2 + sl;
    %%%%%%%%%%
    temp_Im1(abs(temp_Im1)>1)=1;  %option
    I21 = fft2(temp_Im1);
    I21(mask==1)=f(mask==1);
    Im1 = abs(ifft2(I21));   
    Im1(abs(Im1)>1)=1;  %option
    
    Im1 = abs(ifft2( (f + beta * fft2(Im1)) ./ (M_mask + beta) ));    %beta       =   beta * 1.05;
    
    Im = Im1;
    %% subproblem-3 %%%%%%%%%%%%%%%%%%
    top  = fft2(Im);
    for iiiii = 1:3
        bottom = 1;
        for jjj = 1:size(F1,4)
            Weight_mu2 = 0.00001 + abs( conj(filter_eigsK(:,:,jjj)) .* fft2(Im) ).^2;
            bottom = bottom + filter_mu2 * (1./Weight_mu2).* abs(filter_eigsK(:,:,jjj)).^2;
        end
        Im = abs(ifft2( top ./  bottom ));
    end
    
    %%%%%%%%%%%%% save PSNR/HFEN %%%%%%%%%%%%%%%
    %figure(300);imagesc(abs(Im));colormap(gray);axis off; axis equal;  
    %figure(333);imagesc(abs(Im/255-M0/255));axis off; axis equal;colorbar;
    highfritererror(iteri)=norm(imfilter(abs(Im)/max(abs(Im(:))),fspecial('log',15,1.5)) - imfilter(M0/max(M0(:)),fspecial('log',15,1.5)),'fro');
    psnr_idex(iteri) = psnr(Im,M0,1);
    disp(sprintf('iteri: %d, PSNR: %4.2f ', iteri, psnr_idex(iteri)));  %
end;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%% step3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(70000); plot(psnr_idex,'rv-');
param2.Im=Im;
param2.PSNR=psnr_idex;
param2.HFEN=highfritererror;
%%%%%%%%%%%%%%%%%%%%%%%%%
dictionary_temp  = reshape(F1(:,:,1,:),[size(F1,1),size(F1,2),size(F1,4)]);
d =  reshape(dictionary_temp,[size(F1,1)*size(F1,2),size(F1,4)]);
dictionary_temp = plane_normalize(dictionary_temp);
figure(12901);subplot(2,1,2);II = displayDictionary_nonsquare2RC(d,2,9,0);  %
