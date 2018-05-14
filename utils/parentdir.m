%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Returns the parent directory of a path (which could be a file or folder
% itself).
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @fileman_file @copybrief parentdir.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief parentdir.m
%
% @param path to a file or folder
% @retval fullpath the folder the file is in or the parent to the input folder.
% @retval foldername only the name part of the parent folder.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fullpath,foldername] = parentdir(path)

% If path is a file, then it returns the folder the file is in.
% If path is a directory, then it returns the parent folder.

if(isempty(path))
    fullpath = '';
    foldername = '';
else

% Remove the last slash.
if(strcmp(path(end),'/'))
    path = path(1:end-1);
end
% if(strcmp(path(1),'/'))
%     path = path(2:end);
% end



if(isdir(path))
    
    [fullpath,b,c] = fileparts(path);
    if(strcmp(fullpath,'/') || strcmp(fullpath,'.') || strcmp(fullpath,''))
        fullpath = pwd;
    end
    [a,foldername,c] = fileparts(fullpath);
    
else
    path = remove_dot_mat(path);
    [fullpath,b,c] = fileparts(path);
    if(strcmp(fullpath,'/') || strcmp(fullpath,'.') || strcmp(fullpath,''))
        fullpath = pwd;
    end
    [a,foldername,c] = fileparts(fullpath);
end

end


end