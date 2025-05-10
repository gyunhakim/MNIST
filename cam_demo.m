%% -------------------------------------------------
%  (A) 이미 학습된 net 불러오기 (예: net이 있다고 가정)
% --------------------------------------------------
% load('cnn_weights.mat');  % 가중치만 있다면 복원 코드 필요.
% 또는 이전 코드에서 이미 trainNetwork()로 net을 얻었다면 생략 가능.

%% -------------------------------------------------
%  (B) 웹캠 연결
% --------------------------------------------------
cam = webcam;  % 연결된 기본 웹캠 객체
% camList = webcamlist;  % 여러대일 경우 목록 확인

% 웹캠 초기 프레임
frame = snapshot(cam); 
fprintf('웹캠 연결 성공! 프레임 크기: [%d x %d x %d]\n',...
    size(frame,1), size(frame,2), size(frame,3));

%% -------------------------------------------------
%  (C) 실시간 영상 받아서 MNIST 분류 (이진화, 반전, 정규화 고려)
% --------------------------------------------------
figure('Name','Real-Time MNIST Recognition','NumberTitle','off');

while ishandle(gcf)  % Figure 창이 열려있는 동안 반복
    % 1) 웹캠에서 한 프레임을 가져옴
    frame = snapshot(cam);

    % 2) 원본 프레임 표시
    subplot(1,2,1);
    imshow(frame);
    title('Original Webcam Frame');

    % 3) 그레이스케일 + 28x28 리사이즈
    grayFrame = rgb2gray(frame);
    resizedFrame = imresize(grayFrame, [28 28]);

    % -------- [추가 팁 2번: 이진화 + 반전] --------
    % MNIST는 배경=검정(0), 숫자=흰색(1)에 가까우므로, 
    % 웹캠 영상이 반대일 경우 대비해 이진화 & 반전 시도
    bwFrame = imbinarize(resizedFrame);   % 이진화 (0~1, logical)
    bwFrame = imcomplement(bwFrame);      % 반전

    % -------- [추가 팁 3번: 정규화] --------
    % 학습 당시 별도 정규화를 안 했다면 생략 가능하지만,
    % 조명 차이 대응을 위해 예시로 평균/표준편차 정규화
    % (아래 meanVal, stdVal은 임의 예시)
    meanVal = 0.1307;  % MNIST grayscale 평균(대략 PyTorch 기준 예시)
    stdVal  = 0.3081;  % MNIST grayscale 표준편차(대략)
    X = single(bwFrame);                % float 변환 (0 or 1)
    X = (X - meanVal) / stdVal;         % 정규화
    % ---------------------------------------

    % CNN 입력 차원 맞춤 [28 x 28 x 1 x 1]
    input4D = reshape(X, [28 28 1 1]);

    % 4) CNN 분류
    YPred = classify(net, input4D);

    % 5) 이진화 후 영상도 표시
    subplot(1,2,2);
    imshow(bwFrame);
    title(['Predicted Digit: ', char(YPred)], 'FontSize', 14, 'Color','b');
    drawnow;

    pause(0.1);
end

% --------------------------------------------------
% 반복문 종료 시 webcam 해제
clear cam;
disp('웹캠 실시간 인식 종료');
