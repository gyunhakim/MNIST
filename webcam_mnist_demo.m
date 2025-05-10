function webcam_mnist_demo()
    %---------------------------------------------------------------
    % 1) 학습 완료된 가중치 불러오기
    %    (mnist_sigmoid_bp_weights.mat 파일에 W1, b1, W2, b2 저장)
    %---------------------------------------------------------------
    weightsFile = 'mnist_sigmoid_fir_weights.mat';
    if ~isfile(weightsFile)
        error('파일 %s 이(가) 존재하지 않습니다. 먼저 학습을 진행해주세요.', weightsFile);
    end
    load(weightsFile, 'W1', 'b1', 'W2', 'b2');  % 로드

    %---------------------------------------------------------------
    % 2) 웹캠 연결 (MATLAB에서 Support Package for USB Webcam 필요)
    %---------------------------------------------------------------
    try
        cam = webcam;  % 웹캠 객체 생성(기본 장치)
    catch ME
        disp(ME.message);
        error('웹캠을 열 수 없습니다. MATLAB Support Package 설치/장치 연결 확인 요망.');
    end

    %---------------------------------------------------------------
    % 3) Sigmoid 함수 정의 (학습 시와 동일하게)
    %---------------------------------------------------------------
    sigmoid = @(x) 1./(1+exp(-x));  % 학습 코드에서 Z1계산 시 -2.5*x 를 썼다면 맞춰줘야 함
    % 학습 코드에서는 "Z1 = X*W1 + b1" 후에 "sigmoid(Z1) = 1./(1+exp(-2.5*Z1))" 형태였으므로
    % 여기서도 -2.5를 맞춰주거나, 학습 코드의 sigmoid와 동일한 형태로 정의해야 합니다.
    % 아래에서는 "학습 코드의 evaluate_nn()"를 그대로 가져온다고 가정하겠습니다.

    %---------------------------------------------------------------
    % 4) N회 반복 촬영하여 예측
    %---------------------------------------------------------------
    numIterations = 1000;  % 원하는 횟수
    figure('Name','MNIST Webcam Demo','NumberTitle','off');
    for iter = 1:numIterations
        % (a) 웹캠 이미지 캡처
        img = snapshot(cam);

        % (b) 흑백 변환 & 28×28 리사이즈
        imgGray = rgb2gray(img);
        imgGrayInv = imcomplement(imgGray);
        imgResized = imresize(imgGrayInv, [28 28]);

        % (c) 784차원(1×784) 벡터 & 0~1 정규화
        Xtest = double(reshape(imgResized, 1, 28*28)) / 255.0;

        % (d) Forward: MLP (학습 코드의 evaluate_nn()와 동일하게)
        %     Z1 = X*W1 + b1;  A1 = sigmoid(Z1*α);  Z2 = A1*W2 + b2;  softmax => label
        %     *중요: 학습 때 sigmoid가 1./(1+exp(-2.5*z)) 라면 동일하게 2.5 배수 고려*
        Z1 = Xtest * W1 + b1;  
        A1 = 1./(1+exp(-Z1));  % 학습 코드에서 -2.5를 쓰셨다면 여기도 동일하게!
        Z2 = A1 * W2 + b2;
        expZ = exp(Z2);
        Y_hat = expZ ./ sum(expZ, 2);  % 소프트맥스

        [~, predClass] = max(Y_hat, [], 2);  % 1..10
        predClass = predClass - 1;           % 0..9

        % (e) 결과 출력
        subplot(1,2,1);
        imshow(img);  % 원본 컬러이미지 표시
        title('Captured Image');

        subplot(1,2,2);
        imshow(imgResized);  % 28x28 리사이즈된 grayscale
        title(sprintf('Resized 28x28 | 예측: %d', predClass));

        drawnow;
        pause(1);  % 2초 대기(원하는 만큼 조절)
    end

    %---------------------------------------------------------------
    % 5) 웹캠 릴리즈
    %---------------------------------------------------------------
    clear cam;
    disp('웹캠 데모 종료');
end
