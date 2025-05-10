clear; clc; close all;

%% (A) MNIST 로드
load('mnist.mat','train_images','train_labels','test_images','test_labels');
% 가정: train_images [28 x 28 x 60000], test_images [28 x 28 x 10000]
Ntrain = size(train_images,3);
Ntest  = size(test_images,3);

% CNN 입력으로 쓰기 위해 4차원 [28 x 28 x 1 x N] 형태로 reshape
train_images_cnn = reshape(train_images, [28, 28, 1, Ntrain]);  % [28 x 28 x 1 x 60000]
test_images_cnn  = reshape(test_images,  [28, 28, 1, Ntest]);   % [28 x 28 x 1 x 10000]

% 클래스 레이블을 categorical로 변환
train_labels_cat = categorical(train_labels);
test_labels_cat  = categorical(test_labels);

%% (B) CNN 구성
% conv1 -> bn1 -> relu1 -> pool1
% conv2 -> bn2 -> relu2 -> pool2
% fc -> softmax -> classificationLayer
layers = [
    imageInputLayer([28 28 1], 'Name','input','Normalization','none')

    convolution2dLayer(3, 16, 'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3, 32, 'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    fullyConnectedLayer(10,'Name','fc')  % MNIST 10클래스
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];

%% (C) 학습 옵션 설정
% - 'VerboseFrequency'를 지정하면 Iteration마다 Command Window에 Loss가 표시됨.
% - 'Plots','training-progress'는 Iteration 단위로 실시간 그래프를 띄워줌.
options = trainingOptions('sgdm', ...
    'MaxEpochs', 2, ...              % 예) 5 epoch
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.01, ...
    'Plots','training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 1);          % 1이면 모든 iteration마다 표시 (너무 자주면 크게 해도 됨)

%% (D) 학습
[net, info] = trainNetwork(train_images_cnn, train_labels_cat, layers, options);

%% (E) 테스트 정확도
YPred = classify(net, test_images_cnn);
acc = mean(YPred == test_labels_cat);
fprintf('Test Accuracy: %.2f%%\n', acc*100);

%% (F) "에폭별" 평균 Loss 계산하여 그래프 그리기
% info.TrainingLoss는 iteration(미니배치) 기준으로 기록됨.
%  => 한 epoch = (Ntrain / MiniBatchSize)번의 iteration
% 실제 iteration 수
numIterations = numel(info.TrainingLoss);

% 옵션에서 설정한 epoch 수
numEpochs = options.MaxEpochs;

% epoch당 iteration 수 (정수로 맞지 않을 수도 있으므로 round 처리)
itPerEpoch = round(numIterations / numEpochs);

epochLoss = zeros(numEpochs,1);
for e = 1:numEpochs
    % iteration 구간 추출
    idxStart = (e-1)*itPerEpoch + 1;
    idxEnd   = min(e*itPerEpoch, numIterations);  % 마지막 epoch은 범위 넘어가면 컷
    epochLoss(e) = mean(info.TrainingLoss(idxStart:idxEnd));
end

figure;
plot(1:numEpochs, epochLoss, 'b-o','LineWidth',1.5);
xlabel('Epoch');
ylabel('Loss');
title('CNN Training: Epoch vs. Loss');
grid on;

%% (G) 최종 가중치 저장하기 (conv1, conv2, fc)
% layerGraph:
%   1: input
%   2: conv1, 3: bn1, 4: relu1, 5: pool1
%   6: conv2, 7: bn2, 8: relu2, 9: pool2
%   10: fc, 11: softmax, 12: classoutput

conv1W = net.Layers(2).Weights;  % conv1 필터
conv1b = net.Layers(2).Bias;
conv2W = net.Layers(6).Weights;  % conv2 필터
conv2b = net.Layers(6).Bias;
fcW    = net.Layers(10).Weights; % fc 가중치
fcB    = net.Layers(10).Bias;

save('cnn_weights.mat','conv1W','conv1b','conv2W','conv2b','fcW','fcB','-v7');
fprintf('Saved final weights => cnn_weights.mat\n');

