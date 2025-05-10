    clear; clc; close all;

    %% (A) MNIST 로드 & 전처리
    load('mnist.mat','train_images','train_labels','test_images','test_labels');

    % 이미지 reshape: (28*28) -> 열벡터 / 샘플별 행 (60000×784)
    train_images = double(reshape(train_images, 28*28,[])');
    test_images  = double(reshape(test_images,  28*28,[])');
    train_labels = double(train_labels);  % 0..9
    test_labels  = double(test_labels);   % 0..9

    %% ----------------------------------------------
    %  (1) 훈련 샘플 개수 (subset_size) 지정
    %% ----------------------------------------------
    subset_size = 60000;  % 원하는 개수로 설정 (최대 60000)
    subset_size = min(subset_size, size(train_images,1));
    train_images = train_images(1:subset_size,:);
    train_labels = train_labels(1:subset_size);

    % 최종 훈련 세트 크기
    [Ntrain, input_dim] = size(train_images);
    fprintf('Training set: %d samples (subset), input_dim=%d\n', Ntrain, input_dim);

    %% (A-2) One-hot 라벨
    num_classes = 10;
    Y_train_onehot = zeros(Ntrain, num_classes);
    for i = 1:Ntrain
        Y_train_onehot(i, train_labels(i)+1) = 1;
    end

    % 테스트 세트도 reshape
    Ntest = size(test_images,1);
    Y_test_onehot = zeros(Ntest, num_classes);
    for i=1:Ntest
        Y_test_onehot(i, test_labels(i)+1) = 1;
    end

    %% (B) 네트워크 구조 (1 hidden layer, sigmoid)
    hidden_dim = 128;
    output_dim = 10;
    rng('default');
    rng(0, 'twister');  
    W1 = 0.01*randn(input_dim, hidden_dim);
    b1 = zeros(1, hidden_dim);
    W2 = 0.01*randn(hidden_dim, output_dim);
    b2 = zeros(1, output_dim);

    %% (C) 학습 하이퍼파라미터
    epochs = 10;
    batch_size = 64;
    learning_rate = 0.01;
    num_batches = floor(Ntrain/batch_size);

    % 학습 중 Loss 기록용
    loss_history = zeros(epochs,1);

    % Sigmoid 함수
    sigmoid = @(x) 1./(1+exp(-x));

    %% (D) 학습 루프 (BP)
    for epoch = 1:epochs
        perm = randperm(Ntrain);

        epoch_loss_sum = 0;

        for nb = 1:num_batches
            idx = perm((nb-1)*batch_size+1 : nb*batch_size);

            Xb = train_images(idx,:);
            Yb = Y_train_onehot(idx,:);

            %--- Forward ---
            Z1 = Xb*W1 + b1;         
            A1 = sigmoid(Z1);       
            Z2 = A1*W2 + b2;        

            %--- Softmax ---
            expZ = exp(Z2);
            Y_hat = expZ ./ sum(expZ,2);

            %--- CrossEntropy ---
            ce = -sum( Yb .* log(max(Y_hat,1e-8)), 2);  
            batch_loss = mean(ce);                      
            epoch_loss_sum = epoch_loss_sum + batch_loss;

            %--- Backprop ---
            dZ2 = (Y_hat - Yb) / batch_size;  
            gradW2 = A1' * dZ2;               
            gradb2 = sum(dZ2,1);

            dA1 = dZ2 * W2';                  
            dZ1 = dA1 .* (A1 .* (1-A1));  

            gradW1 = Xb' * dZ1;
            gradb1 = sum(dZ1,1);

            %--- Update ---
            W2 = W2 - learning_rate * gradW2;
            b2 = b2 - learning_rate * gradb2;
            W1 = W1 - learning_rate * gradW1;
            b1 = b1 - learning_rate * gradb1;
        end

        epoch_loss_mean = epoch_loss_sum / num_batches;
        loss_history(epoch) = epoch_loss_mean;

        fprintf('[Epoch %2d/%2d] Loss=%.4f\n', epoch, epochs, epoch_loss_mean);
    end

    %% (E) 테스트 정확도
    [acc, ~] = evaluate_nn(test_images, test_labels, W1, b1, W2, b2);
    fprintf('Test Accuracy: %.2f%%\n', acc*100);

    %% (F) 가중치 저장
    save('mnist_sigmoid_bp_weights.mat','W1','b1','W2','b2','-v7');
    fprintf('Saved final weights => mnist_sigmoid_bp_weights.mat\n');

    %% (G) ONNX 내보내기
    net_onnx_export_sigmoid(W1,b1,W2,b2, hidden_dim, output_dim);

    %% (H) Epoch vs. Loss 그래프
    figure;
    plot(1:epochs, loss_history, 'b-o','LineWidth',1.5);
    xlabel('Epoch');
    ylabel('Loss');
    title(['MLP Training: Epoch vs. Loss (subset=' num2str(subset_size) ')']);
    grid on;



%%%%%%%%% 보조함수1: 테스트 정확도 평가 %%%%%%%%%
function [acc, preds] = evaluate_nn(X, labels, W1, b1, W2, b2)
    Z1 = X*W1 + b1;
    A1 = 1./(1 + exp(-Z1));
    Z2 = A1*W2 + b2;
    expZ = exp(Z2);
    Y_hat = expZ ./ sum(expZ,2);

    [~, pred_idx] = max(Y_hat,[],2); 
    preds = pred_idx - 1;           
    acc = mean(preds == labels);
end


%%%%%%%%% 보조함수2: ONNX 내보내기 (동일) %%%%%%%%%
function net_onnx_export_sigmoid(W1,b1,W2,b2, hidden_dim, output_dim)
    lgraph = layerGraph([
        imageInputLayer([28 28 1],'Name','input','Normalization','none')
        fullyConnectedLayer(hidden_dim,'Name','fc1')
        fullyConnectedLayer(output_dim, 'Name','fc2')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput','Classes',categorical(0:9))
    ]);

    % fc1 가중치 주입
    fc1_new = fullyConnectedLayer(hidden_dim,'Name','fc1');
    fc1_new.Weights = W1';
    fc1_new.Bias    = b1';
    lgraph = replaceLayer(lgraph, 'fc1', fc1_new);

    % fc2 가중치 주입
    fc2_new = fullyConnectedLayer(output_dim,'Name','fc2');
    fc2_new.Weights = W2';
    fc2_new.Bias    = b2';
    lgraph = replaceLayer(lgraph, 'fc2', fc2_new);

    netTemp = assembleNetwork(lgraph);
    onnx_filename = "mnist_sigmoid_bp.onnx";
    exportONNXNetwork(netTemp, onnx_filename, 'OpsetVersion',12);
    fprintf('Exported ONNX => %s\n', onnx_filename);
end
