# Car_blackbox_Collision_

1. **프로젝트 이름** : 자동차 충돌 분석

3. **프로젝트 선정 동기** : 이 프로젝트를 선택한 것에는 두가지 이유가 있다. 첫번째는 이번 학기동안 *opencv*를 사용해 영상들을 어떻게 다루는지 배웠다. 이에 영상 데이터를 다뤄보고 싶었고 opencv 라이브러리를 사용해 프로젝트를 진행하고 싶었다. 두번째는 최근 블랙박스를 활용하여 도로 위에서의 사고를 예방하는 연구가 다양하게 진행되고 있다고 한다. 근래 블랙박스는 위험 상황을 스스로 판단하고, 실시간으로 대처하는 방향으로 발전하고 있어 교통사고의 발생과 규모를 큰 폭으로 감소시킨다고 한다. 이렇게 선택한 프로젝트가 블랙박스 영상 classification이다. 주제 선정에서의 아쉬운 점은 프로젝트를 진행하면서 dataset을 구성할때 빼고는 opencv를 다루지 않았다는 점이다.

4. **프로젝트 task** : Classification

5. **사용할 데이터셋에 대한 설명** : 

    1. 학습용 차량 블랙박스 영상은 총 2698개이다.
    
    1. 영상 데이터는 10개의 프레임으로 되어 있으며 5초 분량이다.
    
    1. 주어진 csv파일에는 영상 이름과 영상이 담긴 주소 경로, 예측할 class가 담겨있다.
        - ![image](https://github.com/Seeooo-0/Car_blackbox_Collision_/assets/90232567/9084370b-eec7-4b76-82b6-2bae3de6009f)
    
    1. class는 총 13개로 아래와 같다.
        - ![1675581601829146](https://github.com/Seeooo-0/Car_blackbox_Collision_/assets/90232567/f0558f52-5f67-4203-a679-263caea3e8cb)
    
    1. 데이터셋을 확인해 보면 데이터 개수가 많이 부족하고 imbalance하다는 것을 알 수 있었다.
        - 이에 영상 augmentation
        - imbalance 데이터셋에 robust한 Loss함수 사용
        - 각 class별 weights 계산
        - UnderSampling 또는 Oversampling 가 필요 할 것이라고 생각했다.

6. **관련 연구**
    1. r3_18 (torchvision의 video모델)
    2. mc3_18 (torchvision의 video모델)
    3. swind3d_s (torchvision의 video모델)
    4. slow_r50 (facebookresearch 모델)
    5. slowfast_r50 (facebookresearch 모델)
    6. x3d_s (facebookresearch 모델)
    - main코드에선 r3_18 모델을 사용했다.
    
7. **데이터 전처리**
    1. upload한 코드에서 `video_visualization.ipynb` 파일을 확인하면 된다.
    1. 코드와 같이 labeling을 다시 진행했는데, 그 이유는 crash와 non-crash 데이터 사이의 차이가 매우 크기 때문이다.

8. **Loss function**
    1. class imbalance에 robust한 로스는 크게 두가지라 한다.
        - Focal loss
        - Asymmetric loss
        - main 코드에서는 후자를 사용했다.

9. **평가 지표**
    1. F1-score
        - ![image](https://github.com/Seeooo-0/Car_blackbox_Collision_/assets/90232567/ea3c7c3b-528d-4ce6-9279-67ead6afc40f)

10. **훈련**
    1. 훈련 환경
        - 처음엔 영상 데이터라 메모리 부족 현상이 잃어나 RTX 2080 Ti로 batch size를 4로 baseline 모델 훈련을 진행했다.
        - 실제 훈련 시에는 A100-SXM-80GB 로 진행했다. 한 에폭당 train과 validation 합쳐 16~7분 소요됐다.
    1. 훈련 결과
        - train loss
            - ![train_loss](https://github.com/Seeooo-0/Car_blackbox_Collision_/assets/90232567/06a25302-af21-490e-9e6b-83bfcbfaaea9)
        - validation loss
            - ![validation_loss](https://github.com/Seeooo-0/Car_blackbox_Collision_/assets/90232567/851aeb4b-65da-43b6-b0c1-e1c7b1fe2000)
        - f1 score
            - ![f1 score](https://github.com/Seeooo-0/Car_blackbox_Collision_/assets/90232567/edd9dfae-ba19-40ca-841b-09ff609ee24b)
        
11. **결과**
    - 앞서 훈련한 3가지의 모델을 inference시에 합쳐서 다시 1~13까지의 라벨로 분류하도록 진행했다.
    - 10에폭식 학습을 진행하는데 마지막 10에폭에서 loss 값이 이상하게 발산하는 것을 확인할 수 있었다. 갑작스럽게 한 에폭에서만 확인되는 경향으로 보아선 코드 부분에 이상이 있는 것으로 보이지만 오류를 찾진 못했다. 다시 진행한다면 9에폭까지만 진행해야 할 것 같다..
    - 프로젝트를 진행하면서 아쉬운 점은 video augmentation을 진행하고 싶어 여러가지 라이브러리로 진행해보았다. albumentations, create_video_transform 등 이 있었지만 3가지의 라이브러리 모두 진행에 있어 오류가 나와 transform은 만들어 주지 못했다. 대신 customdataset 자체에서 정규화와 resize만 진행했다.
    - 또한 dataset의 labeling에 있어 오류가 많아 좋은 성능을 내기 위해선 데이터를 하나하나 확인해가며 labeling을 새로 해줘야 하는 상황이였는데, 그 부분은 진행상 여건이 안돼 하지 못한 점이 아쉽다.
    - 이번 프로젝트를 진행하면서 wandb와 연동해서 plot을 뽑아보며 많이 사용하는 tool을 다루는 좋은 기회가 되었다.

    
    
