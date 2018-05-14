%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Removes the \c .mat extension (if one exists) of the input filepath.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief remove_dot_mat.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief remove_dot_mat.m
%
% @param path to a file.
% @retval path removes the \c .mat extension if one exists.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [path] = remove_dot_mat(path)
% Takes in a file path
% Removes the '.mat' file extension (if it exists)
% Returns the path.

matpath = str2mat(path);
if(strcmp(matpath(end-3:end),'.mat'))
    path = matpath(1:end-4);
end

end