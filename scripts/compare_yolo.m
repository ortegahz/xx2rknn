%%
close all;

%%
path_img = '/home/manu/tmp/126_1152_576.bmp';

path_txt_ref = '/home/manu/tmp/onnx_output_2.txt';
path_txt_target = '/home/manu/tmp/rk_yolov5_pose_output36-18x27x1.txt';

th = 0.3;
s = 32;
sa = 1;

nb = 1;
na = 3;
no = 9;

%%
img = imread(path_img);
figure; imshow(img);

[imgsz_h, imgsz_w, ~] = size(img);

ny = imgsz_h / s;
nx = imgsz_w / s;

out_ref = load(path_txt_ref);

% reverse order of python numpy
feature_ref = reshape(out_ref, nx, ny, no, na, nb);
feature_ref = permute(feature_ref, [5 4 3 2 1]);  % nb, na, no, ny, nx
feature_ref_show = feature_ref(1, sa, 5, :, :);
feature_ref_show = squeeze(feature_ref_show);
feature_ref_show_sigmoid = 1 ./ (1 + exp(-feature_ref_show));
[r, c, ~] = find(feature_ref_show_sigmoid > th);
hold on; 
plot(c*s, r*s, '.g', 'MarkerSize', 20);
disp('feature_ref_show');
for i =  1 : length(r)
    text(c(i)*s, r(i)*s, ...
        num2str(feature_ref_show_sigmoid(r(i), c(i))), ...
        'color','g');
end
hold off;

%%
out_target = load(path_txt_target);

% reverse order of python numpy
feature_target = reshape(out_target, nx, ny, no, na, nb);
feature_target = permute(feature_target, [5 4 3 2 1]);  % nb, na, no, ny, nx
feature_target_show = feature_target(1, sa, 5, :, :);
feature_target_show = squeeze(feature_target_show);
feature_target_show_sigmoid = 1 ./ (1 + exp(-feature_target_show));
[r, c, ~] = find(feature_target_show_sigmoid > th);
hold on; 
plot(c*s, r*s, '.r', 'MarkerSize', 20);
disp('feature_target_show');
for i =  1 : length(r)
    text(c(i)*s, r(i)*s, ...
        num2str(feature_target_show_sigmoid(r(i), c(i))), ...
        'color','r');
end
hold off;

