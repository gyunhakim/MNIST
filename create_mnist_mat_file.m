function create_mnist_mat_file()
% CREATE_MNIST_MAT_FILE 
%   - MNIST 공식 바이너리 파일(ubyte 포맷)을 읽어들여
%   - train_images, train_labels, test_images, test_labels 변수를 만든 후
%   - mnist.mat 파일로 저장한다.
%
% 필요한 파일(동일 폴더 내):
%   train-images-idx3-ubyte
%   train-labels-idx1-ubyte
%   t10k-images-idx3-ubyte
%   t10k-labels-idx1-ubyte
%
% (c) ChatGPT Example Code

    % 혹시 .gz 파일만 있고 .ubyte 파일이 없는 경우, 아래 코드로 압축해제 가능
    gunzip('train-images-idx3-ubyte.gz');
    gunzip('train-labels-idx1-ubyte.gz');
    gunzip('t10k-images-idx3-ubyte.gz');
    gunzip('t10k-labels-idx1-ubyte.gz');

    % 1) 학습 이미지 (28x28x60000) 로드
    [train_images, num_train_img] = load_images('train-images-idx3-ubyte');
    fprintf('train_images loaded: %d images\n', num_train_img);   % 60000

    % 2) 학습 레이블 (60000x1) 로드
    train_labels = load_labels('train-labels-idx1-ubyte');
    fprintf('train_labels loaded: %d labels\n', length(train_labels));

    % 3) 테스트 이미지 (28x28x10000) 로드
    [test_images, num_test_img] = load_images('t10k-images-idx3-ubyte');
    fprintf('test_images loaded: %d images\n', num_test_img);     % 10000

    % 4) 테스트 레이블 (10000x1) 로드
    test_labels = load_labels('t10k-labels-idx1-ubyte');
    fprintf('test_labels loaded: %d labels\n', length(test_labels));

    % 최종 저장
    save('mnist.mat', 'train_images', 'train_labels', 'test_images', 'test_labels');
    fprintf('Saved to mnist.mat\n');
end


% =============================
% [A] 이미지 파일 로더
% =============================
function [images, numImages] = load_images(filename)
    fid = fopen(filename, 'rb');
    if fid < 0
        error('Cannot open file: %s', filename);
    end

    % IDX 파일 헤더 참고
    %  - Magic number (4 bytes) : 2051 (0x00000803)
    %  - Number of images (4 bytes)
    %  - Number of rows   (4 bytes) : 28
    %  - Number of columns(4 bytes) : 28
    magic = fread(fid, 1, 'int32', 'b');  % big-endian
    if magic ~= 2051
        error('Invalid magic number in %s. Expected 2051, got %d', filename, magic);
    end

    numImages = fread(fid, 1, 'int32', 'b');
    numRows   = fread(fid, 1, 'int32', 'b');
    numCols   = fread(fid, 1, 'int32', 'b');

    % 이미지 픽셀 읽기: unsigned byte (0~255)
    % 전체 크기: numImages x numRows x numCols
    imageData = fread(fid, numImages*numRows*numCols, 'uint8');
    fclose(fid);

    % 28x28xN 형태로 reshape
    images = reshape(imageData, [numRows, numCols, numImages]);

    % float/double 변환 & [0,1] 정규화(선택)
    images = double(images) / 255.0;
end


% =============================
% [B] 레이블 파일 로더
% =============================
function labels = load_labels(filename)
    fid = fopen(filename, 'rb');
    if fid < 0
        error('Cannot open file: %s', filename);
    end

    % IDX 파일 헤더 참고
    %  - Magic number (4 bytes) : 2049 (0x00000801)
    %  - Number of labels (4 bytes)
    magic = fread(fid, 1, 'int32', 'b');
    if magic ~= 2049
        error('Invalid magic number in %s. Expected 2049, got %d', filename, magic);
    end

    numLabels = fread(fid, 1, 'int32', 'b');

    % 각 레이블: unsigned byte (0~9)
    labelData = fread(fid, numLabels, 'uint8');
    fclose(fid);

    labels = double(labelData);  % [numLabels x 1], 범위 0~9
end
