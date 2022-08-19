%%
clear; close all;

%%
% out_ref = load('/home/manu/tmp/resnet50v2_rknn.txt');
% out_target = load('/home/manu/tmp/resnet50v2_onnx.txt');
out_ref = load('/home/manu/tmp/feat_rknn.txt');
out_target = load('/home/manu/tmp/feat_onnx.txt');

%%
out_all = cat(1, out_ref, out_target);

error = sum((out_ref - out_target) .^ 2) / length(out_ref);
error = error / (max(out_all) - min(out_all));

%%
figure;
subplot(3,1,1);
plot(1:length(out_ref), out_ref, 'g', 1:length(out_target), out_target, 'r')
title(['error: ' num2str(error)]);

subplot(3,1,2); plot(1:length(out_target), out_target, 'r'); title('target');
subplot(3,1,3); plot(1:length(out_ref), out_ref, 'g'); title('ref');

