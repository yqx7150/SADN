%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Converts any old fields in the model structure to new ones so that they are
% compatible with new changes to the code. This checks, USE_IPP, ZERO_MEAN, and
% DISPLAY_ERRORS fields.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @other_comp_file @copybrief backwards_compatible.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief backwards_compatible.m
%
% @param model the model structure that potentially has old fields.
% @retval model the new model structure that has the new fields included.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [model] = backwards_compatible(model)

% Set if you want to use IPP libraries. (MUCH SLOWER IF NOT USED).
if(~isfield(model,'USE_IPP'))
    if(strcmp(model.machine,' laptop'))
        fprintf('Setting USE_IPP to 0 as backwards compatible default.\n');
        model.USE_IPP = 0;
    else
        fprintf('Setting USE_IPP to 1 as backwards compatible default.\n');
        model.USE_IPP = 1;
    end
    if(~exist(strcat('ipp_conv2.',mexext),'file'))
        fprintf('You do not have the compiled versions of the IPP Convolutions Toolbox therefore reverting to slower MATLAB only implementation.\n')
        model.USE_IPP = 0;
    end
end

% Set if you want to subtract the mean from the images before processing.
if(~isfield(model,'ZERO_MEAN'))
    if(isfield(model,'VARIANCE_THRESHOLD'))
        if(model.VARIANCE_THRESHOLD == 1)
            fprintf('Setting ZERO_MEAN to 1 as backwards compatible default.\n');
            model.ZERO_MEAN = 1;
        else
            fprintf('Setting ZERO_MEAN to 0 as backwards compatible default.\n');
            model.ZERO_MEAN = 0;
        end
        model = rmfield(model,'VARIANCE_THRESHOLD');
    else
        fprintf('Setting ZERO_MEAN to 1 as backwards compatible default.\n');
        model.ZERO_MEAN = 1;
    end
    
end


% Make it backwards compatible.
if(~isfield(model,'DISPLAY_ERRORS'))
    fprintf('Setting DISPLAY_ERRORS to 1 as backwards compatible default.\n');
    model.DISPLAY_ERRORS = 1;
end





end