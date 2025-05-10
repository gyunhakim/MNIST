%% mnist_ff_bp_step1.m
% MATLAB에서 MNIST로 Feedforward(MLP) 신경망을 학습하는 예시 코드
% (단계 1: Backprop 방식으로 네트워크 학습)

clear; clc; close all;

%% 1) MNIST 데이터 불러오기
% mnist.mat 내부에 다음 변수가 있다고 가정:
%   - train_images: [28×28×60000], train_labels: [60000×1] (0~9)
%   - test_images:  [28×28×10000], test_labels : [10000×1]
load('mnist.mat','train_images','train_labels','test_images','test_labels');

% 크기 확인
fprintf('train_images size: %s\n', mat2str(size(train_images)));  % [28 28 60000]
fprintf('train_labels size: %s\n', mat2str(size(train_labels)));  % [60000 1]

% 28×28×(1)×(개수) 형태로 reshape → imageInputLayer 사용 가능
train_images = reshape(train_images, 28,28,1, []);
test_images  = reshape(test_images,  28,28,1, []);

% 라벨을 categorical로 변환
train_labels_cat = categorical(train_labels);  % [60000×1]
test_labels_cat  = categorical(test_labels);   % [10000×1]

%% 2) MLP(Feedforward) 네트워크 정의
% "Flatten" 과정을 자동으로 처리해주는 대신, 
% imageInputLayer([28 28 1]) → fullyConnectedLayer → ReLU ... 반복
% (단순 예시: 은닉층 2개 + 출력 10)

layers = [
    imageInputLayer([28 28 1], 'Name','input')
    
    fullyConnectedLayer(128, 'Name','fc1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(64, 'Name','fc2')
    reluLayer('Name','relu2')
    
    fullyConnectedLayer(10, 'Name','fc3')        % MNIST 10클래스
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')
];

%% 3) 학습 옵션 설정 (SGDM)
options = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.01, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true);

%% 4) 네트워크 학습
net = trainNetwork(train_images, train_labels_cat, layers, options);

% 학습 종료 후 net은 DAGNetwork(또는 SeriesNetwork) 형태

%% 5) 정확도 평가
YPred = classify(net, test_images);
acc = mean(YPred == test_labels_cat);
fprintf('Test Accuracy: %.2f%%\n', acc*100);

%% 6) 모델 저장 (MATLAB 형식)
save('mnist_ff_bp.mat','net');
fprintf('Trained MLP model saved: mnist_ff_bp.mat\n');
