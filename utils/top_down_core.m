%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% The basis of going from the top of the model down to the bottom.
% This is used in train_recon_layer, train_recon_yann, top_down, etc.
% Assumes all variables are set externally and all models are loaded.
% Note: Make sure layer is the top_layer.
% model is assumed to be the top model and contain information about each
% layer below.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @outparam \li \e layer the top layer of the model.
% \li \e recon_z# the top layer's feature maps.
% \li \e pooled_indices# the top layer and below layers indices of Max pooling
% (if used).
% \li \e model# the model struct for each layer containing all parameters of the layers.
% Particularly norm_types, norm_sizes, xdim, ydim, filter_size, z0_filter_size,
% TRAIN_Z0, conmats
% \li \e F# the filters for a given layer.
% \li \e z0# the z0 maps for a given layer.
% \li \e COMP_THREADS the number of computation threads you want to use.
%
% @top_down_file @copybrief top_down_core.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(~exist('USE_IPP','var'))
    % Get the USE_IPP flag from the top layer if it doesn't exist.
%     eval(strcat('USE_IPP = model',num2str(layer),'.USE_IPP'));
end

for lay=layer:-1:1
    % Get the string for the layer.
    slayer = num2str(lay);
    
%     % Here you unpool before proceeding with the reconstructions.
%     switch(model.norm_types{lay+1})
%         case 'Max' % Discrete or Probablistic  Max Pooling
%             % If doing top down, use probablistic max pooling since you
%             % don't have bottom up support.
%             if(strcmp(model.expfile,' top_down') || ...
%                     strcmp(model.expfile,' top_down_sampling'))
%                 % Relies on the top most layer to have the correct
%                 % norm_sizes.
%                 eval(strcat('recon_z',slayer,' = reverse_prob_max_pool(recon_z',slayer,',',...
%                     'pooled_indices',slayer,',',...
%                     'model',num2str(layer),'.norm_sizes{',num2str(lay+1),'},',...
%                     '[model',slayer,'.xdim+model',slayer,'.filter_size(lay)-1 ',...
%                     ' model',slayer,'.ydim+model',slayer,'.filter_size(lay)-1]);'));
%             else
%                 eval(strcat('recon_z',slayer,' = reverse_max_pool(recon_z',slayer,',',...
%                     'pooled_indices',slayer,',',...
%                     'model',num2str(layer),'.norm_sizes{',num2str(lay+1),'},',...
%                     '[model',slayer,'.xdim+model',slayer,'.filter_size(lay)-1 ',...
%                     ' model',slayer,'.ydim+model',slayer,'.filter_size(lay)-1]);'));
%             end
%         case 'Avg' % Average Pooling
%             eval(strcat('recon_z',slayer,' = reverse_avg_pool(recon_z',slayer,',',...
%                 'pooled_indices',slayer,',',...
%                 'model',num2str(layer),'.norm_sizes{',num2str(lay+1),'},',...
%                 '[model',slayer,'.xdim+model',slayer,'.filter_size(lay)-1 ',...
%                 ' model',slayer,'.ydim+model',slayer,'.filter_size(lay)-1]);'));
%         case 'Abs_Avg' % Average Pooling
%             eval(strcat('recon_z',slayer,' = reverse_avg_pool(recon_z',slayer,',',...
%                 'pooled_indices',slayer,',',...
%                 'model',num2str(layer),'.norm_sizes{',num2str(lay+1),'},',...
%                 '[model',slayer,'.xdim+model',slayer,'.filter_size(lay)-1 ',...
%                 ' model',slayer,'.ydim+model',slayer,'.filter_size(lay)-1]);'));
%         case 'None'
%     end
    % Rescale each feature map on the way down to be [-1,1]
    % eval(strcat('recon_z',slayer,' = svm_rescale2(recon_z',slayer,');'));
    
%     if(~USE_IPP)
        %         % If trained with z0 map, then use it to reconstruct.
        %         if(eval(strcat('model',slayer,'.TRAIN_Z0')))
        %             % Reconstruct the layer below.
        %             eval(strcat('recon_z',num2str(lay-1),'=',...
        %                 'ReconstructColorImage_z0(',...
        %                 'z0',slayer,',',...
        %                 'model',slayer,'.z0_filter_size,',...
        %                 'model',slayer,'.num_feature_maps(',slayer','),',...
        %                 'model',slayer,'.num_input_maps(',slayer','),',...
        %                 'recon_z',slayer,',',...
        %                 'F',slayer,',',...
        %                 'model',slayer,'.conmats{',slayer','});'));
        %
        %         else % Not trained with z0
        %             % Reconstruct the layer below.
        %             eval(strcat('recon_z',num2str(lay-1),'=',...
        %                 'ReconstructColorImage(',...
        %                 'model',slayer,'.num_feature_maps(',slayer','),',...
        %                 'model',slayer,'.num_input_maps(',slayer','),',...
        %                 'recon_z',slayer,',',...
        %                 'F',slayer,',',...
        %                 'model',slayer,'.conmats{',slayer','});'));
        %         end
        % Use the new ipp_recon to reconstruct top down.
        eval(strcat('recon_z',num2str(lay-1),'=',...
            'fast_recon(',...
            'z0',slayer,',',...
            'model',slayer,'.z0_filter_size,',...
            'single(recon_z',slayer,'),',...
            'F',slayer,',',...
            'model',slayer,'.conmats{',slayer','},',...
            'model',slayer,'.TRAIN_Z0);'));
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         if lay == 2
%             recon_z1=fast_recon(z02,model2.z0_filter_size,single(recon_z2),F2,model2.conmats{2},model2.TRAIN_Z0);
%         elseif lay == 1
%             recon_z0=fast_recon(z01,model1.z0_filter_size,single(recon_z1),F1,model1.conmats{1},model1.TRAIN_Z0);
%         end        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     else
%         % Use the new ipp_recon to reconstruct top down.
%         eval(strcat('recon_z',num2str(lay-1),'=',...
%             'ipp_recon(',...
%             'z0',slayer,',',...
%             'model',slayer,'.z0_filter_size,',...
%             'single(recon_z',slayer,'),',...
%             'F',slayer,',',...
%             'model',slayer,'.conmats{',slayer','},',...
%             'model',slayer,'.TRAIN_Z0,COMP_THREADS);'));
%     end
end

