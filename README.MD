1차 실험 내용
- MIMIC-CXR 데이터 셋과 ReX 데이터 셋을 함께 사용
- Train:Valid:Test=7:1:2로 변경 --> Train과 Test의 완전한 분리를 위해 Train:Valid=8:2 로 변경 할 수 있으며, external data를 사용하여 테스트
- CosineEmbeddingLoss loss 계산 방식을 sum과 mean 나누어하여 학습. 10epoch이상 학습하였으며, 대략 5epoch부터 과적합 발생
- Aumgent 방식 추가 적용: Image의 pixel 위치 분포를 크게 해치지 않는 colorjitter, gaussianblur, gaussiannoise, gamma를 사용. 추가로, 기존에는 원본 이미지를 resize하였는데, 이는 직사각형의 이미지가 찌그러지는 문제가 발생할 가능성을 고려하여 padding(w,h 중 max로 패딩)하여 정사각형으로 이미지를 구축하여 resize

-- 고려해볼 사항: Embedding 기반의 image finding 시에 free text note의 유사성을 llm으로 점수화