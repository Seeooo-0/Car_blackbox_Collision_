# Car_blackbox_Collision_

1. 프로젝트 이름 : 자동차 충돌 분석

3. 프로젝트 선정 동기 : 이 프로젝트를 선택한 것에는 두가지 이유가 있다. 첫번째는 이번 학기동안 **opencv**를 사용해 영상들을 어떻게 다루는지 배웠다. 이에 영상 데이터를 다뤄보고 싶었고 opencv 라이브러리를 사용해 프로젝트를 진행하고 싶었다. 두번째는 최근 블랙박스를 활용하여 도로 위에서의 사고를 예방하는 연구가 다양하게 진행되고 있다고 한다. 근래 블랙박스는 위험 상황을 스스로 판단하고, 실시간으로 대처하는 방향으로 발전하고 있어 교통사고의 발생과 규모를 큰 폭으로 감소시킨다고 한다. 이렇게 선택한 프로젝트가 블랙박스 영상 classification이다. 주제 선정에서의 아쉬운 점은 프로젝트를 진행하면서 dataset을 구성할때 빼고는 opencv를 다루지 않았다는 점이다.

4. 프로젝트 task : Classification

5. 사용할 데이터셋에 대한 설명 : 

    1. 학습용 차량 블랙박스 영상은 총 2698개이다.
    
    1. 영상 데이터는 10개의 프레임으로 되어 있으며 5초 분량이다.
    
    1. 주어진 csv파일에는 영상 이름과 영상이 담긴 주소 경로, 예측할 class가 담겨있다.
    ![image](https://github.com/Seeooo-0/Car_blackbox_Collision_/assets/90232567/9084370b-eec7-4b76-82b6-2bae3de6009f)
    
    1. class는 총 13개로 아래와 같다.
    ![image](https://dacon.s3.ap-northeast-2.amazonaws.com/competition/236064/editor-image/1675581601829146.jpeg)
    
    1. 데이터셋을 확인해 보면 데이터 개수가 많이 부족하고 imbalance하다는 것을 알 수 있었다.
        - 이에 영상 augmentation
        - imbalance 데이터셋에 robust한 Loss함수 사용
        - 각 class별 weights 계산
        - UnderSampling 또는 Oversampling 가 필요 할 것이라고 생각했다.

6. 관련 연구
    1. r3_18 (torchvision의 video모델)
    2. mc3_18 (torchvision의 video모델)
    3. swind3d_s (torchvision의 video모델)
    4. slow_r50 (facebookresearch 모델)
    5. slowfast_r50 (facebookresearch 모델)
    6. x3d_s (facebookresearch 모델)
    
7. 데이터 전처리
  1. 코드의 
  - 
