%%
% author: manu

%%
close all; clear all;

%%
opts.dir_img = '/home/manu/workspace/rknn/face_recognition/imgs';
opts.output = '/home/manu/workspace/rknn/face_recognition/onnx/dataset.txt';

%%
list_img  = struct2cell(dir(fullfile(opts.dir_img, '*.jpg')))';
paths_img = fullfile(opts.dir_img, list_img(:, 1));

fileID = fopen(opts.output, 'w');

for i = 1 : length(paths_img)
    
    path_img = paths_img{i};
    
    fprintf('processing %dth img %s [total %d]\n', ...
        i, path_img, length(paths_img));

    fprintf(fileID, '%s\n', path_img);

end

fclose(fileID);

%%