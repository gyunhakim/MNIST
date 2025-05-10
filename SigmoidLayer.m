% 파일명: SigmoidLayer.m
% 경로: (mnist_manual_bp_sigmoid.m과 같은 폴더 내)

classdef SigmoidLayer < nnet.layer.Layer
    % 간단한 Sigmoid 활성화 레이어
    %  - predict(~,X) 에서 시그모이드 연산을 수행
    methods
        function layer = SigmoidLayer(name)
            % 생성자: 레이어 이름 설정
            layer.Name = name;
        end

        function Z = predict(~, X)
            % Forward pass에서 사용하는 함수
            % X: (배치 크기 × 채널 × 높이 × 너비) 구조 (실제로는 내부적으로 4D 텐서)
            % 여기서는 elementwise로 1/(1+exp(-X)) 적용
            Z = 1./(1 + exp(-2.5*X));
        end
    end
end
