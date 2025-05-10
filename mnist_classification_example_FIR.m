    clear; clc; close all;

    %% (A) MNIST 로드 & 전처리
    load('mnist.mat','train_images','train_labels','test_images','test_labels');

    train_images = double(reshape(train_images, 28*28,[])');
    test_images  = double(reshape(test_images,  28*28,[])');
    train_labels = double(train_labels);
    test_labels  = double(test_labels);

    %% (A-1) 훈련 샘플 개수 (subset_size)
    subset_size = 6000;  % 원하는대로 조절 (최대 60000)
    subset_size = min(subset_size, size(train_images,1));
    train_images = train_images(1:subset_size,:);
    train_labels = train_labels(1:subset_size);

    [Ntrain, input_dim] = size(train_images);
    fprintf('Training set: %d samples (subset), input_dim=%d\n', Ntrain, input_dim);

    %% (A-2) One-hot 레이블 (출력 10차원)
    num_classes = 10;
    Y_train_onehot = zeros(Ntrain, num_classes);
    for i = 1:Ntrain
        Y_train_onehot(i, train_labels(i)+1) = 1;
    end

    Ntest = size(test_images,1);
    Y_test_onehot = zeros(Ntest, num_classes);
    for i=1:Ntest
        Y_test_onehot(i, test_labels(i)+1) = 1;
    end

    %% (B) 은닉층 (W1,b1)은 고정 (학습X)
    hidden_dim = 128;
    rng('default'); rng(0,'twister');
    W1 = 0.01*randn(input_dim,hidden_dim);
    b1 = zeros(1,hidden_dim);

    % Activation
    sigmoid = @(x) 1./(1+exp(-x));

    %% (C) 출력층 FIR(DTFWNUF) 파라미터
    %   여기서는 편향 b2를 따로 안 쓰고, W2만 업데이트(예시)
    W2 = zeros(hidden_dim, num_classes);

    % FIR 세팅
    N = 6;                % FIR 길이
    Omega = eye(N);       % (N x N)
    lambda = 1e-5;        % 정칙화 파라미터

    %% (D) 학습 하이퍼파라미터
    epochs = 10;
    batch_size = 64;
    num_batches = floor(Ntrain/batch_size);

    % 기록용 (Loss, RMSE)
    loss_history = zeros(epochs,1);
    rmse_history = zeros(epochs,1);

    %% (E) FIR 학습 루프
    for epoch = 1:epochs
        perm = randperm(Ntrain);

        sse_epoch = 0;  % (한 에폭동안의 SSE 누적)
        count_sample = 0;

        for nb = 1:num_batches
            idx_batch = perm((nb-1)*batch_size+1 : nb*batch_size);

            Xb = train_images(idx_batch,:);
            Yb = Y_train_onehot(idx_batch,:);

            for i = 1:batch_size
                x_i = Xb(i,:);
                y_i = Yb(i,:);

                %---------------------------
                % (1) 은닉층 Forward
                %---------------------------
                z1 = x_i*W1 + b1;        % (1 x hidden_dim)
                a1 = sigmoid(z1);       

                %---------------------------
                % (2) FIR 업데이트
                %---------------------------
                store_FIR_data(a1, y_i); 
                FIR_buffer = get_FIR_buffer();
                if size(FIR_buffer.hidden_outs,1) >= N
                    ThetaN = FIR_buffer.hidden_outs(end-N+1:end, :);
                    Yblock = FIR_buffer.labels(end-N+1:end, :);
                    W2 = dtfwnuf_gain_timevary(ThetaN, Yblock, Omega, lambda);
                end

                %---------------------------
                % (3) 출력 및 SSE 누적
                %---------------------------
                out = a1 * W2;   % (1 x 10)
                err = y_i - out; 
                sse_epoch = sse_epoch + sum(err.^2);
                count_sample = count_sample + 1;
            end
        end

        % Epoch별 Loss, RMSE
        %  - Loss = SSE / (총 샘플 수 × 출력 차원)
        %  - RMSE = sqrt(Loss)
        loss_epoch = sse_epoch / (count_sample * num_classes);
        rmse_epoch = sqrt(loss_epoch);

        loss_history(epoch) = loss_epoch;
        rmse_history(epoch) = rmse_epoch;

        fprintf('[Epoch %2d/%2d] Loss=%.4f, RMSE=%.4f\n', ...
            epoch, epochs, loss_epoch, rmse_epoch);
    end

    %% (F) 테스트 정확도
    acc = evaluate_nn_FIR(test_images, test_labels, W1, b1, W2);
    fprintf('Test Accuracy: %.2f%%\n', acc*100);

    %% (G) 결과 그래프
    figure;
    plot(1:epochs, loss_history, 'b-o','LineWidth',1.5); hold on;
    xlabel('Epoch'); ylabel('Loss'); grid on;
    title('FIR Training: Epoch vs. Loss');

    figure;
    plot(1:epochs, rmse_history, 'r-o','LineWidth',1.5); hold on;
    xlabel('Epoch'); ylabel('RMSE'); grid on;
    title('FIR Training: Epoch vs. RMSE');

    %% (H) 학습된 가중치 저장(mat)
    save('mnist_fir_weights.mat','W1','b1','W2','-v7');
    fprintf('Saved final weights => mnist_fir_weights.mat\n');

%% ---------------------------------------------------------------------
%% 보조함수 1) dtfwnuf_gain_timevary (다중출력 + 정칙화)
%% ---------------------------------------------------------------------
function W_new = dtfwnuf_gain_timevary(ThetaN, Yblock, Omega, lambda)
    % ThetaN : (N x hidden_dim)
    % Yblock : (N x num_outputs)
    % Omega  : (N x N)
    % lambda : 정칙화 파라미터

    temp = ThetaN'*(Omega^2)*ThetaN + lambda*eye(size(ThetaN,2));
    G = temp \ (ThetaN'*(Omega^2));
    W_new = G * Yblock;   % (hidden_dim x num_outputs)
end

%% ---------------------------------------------------------------------
%% 보조함수 2) 은닉층 출력을 누적 저장 (최근 N개) - static var 이용
%% ---------------------------------------------------------------------
function store_FIR_data(a1, y_i)
    % call each time with the new (a1,y_i)
    persistent hidden_outs labels
    if isempty(hidden_outs)
        hidden_outs = [];
        labels = [];
    end

    hidden_outs = [hidden_outs; a1];
    labels      = [labels; y_i];

    % overwrite
    assignin('caller','hidden_outs', hidden_outs);
    assignin('caller','labels', labels);
end

function buffer_struct = get_FIR_buffer()
    persistent hidden_outs labels
    if isempty(hidden_outs)
        hidden_outs = [];
        labels = [];
    end
    buffer_struct.hidden_outs = hidden_outs;
    buffer_struct.labels = labels;
end

%% ---------------------------------------------------------------------
%% 보조함수 3) 테스트 정확도 평가 (b2=0 가정)
%% ---------------------------------------------------------------------
function acc = evaluate_nn_FIR(X, labels, W1, b1, W2)
    Z1 = X*W1 + b1;
    A1 = 1./(1+exp(-Z1));
    out = A1*W2;     % (N x 10)

    [~, pred_idx] = max(out, [], 2); 
    preds = pred_idx - 1;   % 0..9
    acc = mean(preds == labels);
end
