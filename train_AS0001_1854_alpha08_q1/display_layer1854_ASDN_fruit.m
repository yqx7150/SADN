clear;clc;
load epoch5_layer1.mat;   
dictionary_temp  = reshape(F(:,:,1,:),[size(F,1),size(F,2),size(F,4)]); 
d =  reshape(dictionary_temp,[size(F,1)*size(F,2),size(F,4)]);
dictionary_temp = plane_normalize(dictionary_temp);
figure(12901);subplot(2,1,2);II = displayDictionary_nonsquare2RC(d,2,9,0);  %




















