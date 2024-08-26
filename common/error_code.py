from enum import Enum


# 에러코드 정의 클래스
class Errors(Enum):
    # module default error
    DEFAULT_ERROR = -1, "(default) Module has been abnormally shut down."

    # 공통/서빙 에러 (700~799)
    E700 = 700, "failed connect DB"
    E701 = 701, "failed insert table"  # 테이블 insert 실패
    E702 = 702, "failed to get ID from DB"
    E703 = 703, "no serving data"  # 서빙 데이터 없음(빈 데이터프레임)
    E704 = 704, "not enough serving data for window size"  # 윈도우 크기에 비해 서빙 데이터 부족함
    E705 = 705, "failed to load model"  # 서빙 모델 로드 실패
    E706 = 706, "no serving data specific columns"  # 서빙 데이터의 특정 컬럼이 없음
    E707 = 707, "serving memory error"  # 서빙 메모리 오류
    E709 = 709, "Instance Create error"  # 인스턴스 생성 오류
    E710 = 710, "Serving Data Trans Dataframe error" # 서빙 데이터 데이터프레임 변환 오류
    E711 = 711, "timeout error"  # 로직 수행시간이 timeout을 초과
    E712 = 712, "logpresso connect param info does not exist"
    E713 = 713, "Request target invalid"
    E714 = 714, "Redis 에 존재하지 않는 모델에 대한 서빙 요청"
    E777 = 777, "unknown serving error"  # unknown serving error

    # 학습 에러 (800~899)
    E800 = 800, "no training data"  # 학습 데이터 없음(빈 데이터프레임)
    E801 = 801, "not enough training data"  # 학습 데이터 부족
    E802 = 802, "prediction features are not provided"  # 사용자가 예측 지표를 설정하지 않음
    E803 = 803, "training features are not provided"  # 사용자가 학습 지표를 설정하지 않음
    E804 = (
        804,
        "feature is not in training dataframe",
    )  # 서버에서 전달해 준 지표가 트레이닝 데이터의 컬럼에 없음
    E805 = 805, "not supported instance type"  # 모듈에서 분석 가능하지 않은 인스턴스를 분석하도록 요청받음
    E806 = 806, "fail to load training dataframe"  # 학습 데이터 로딩 실패
    E807 = (
        807,
        "learning_business_list timestamp is not provided",
    )  # 학습할 비즈니스데이 정보가 입력되지 않음
    E808 = (
        808,
        "timestamp of excpet_business_list or failure_date is not provided",
    )  # 제외할 비즈니스데이 정보가 입력되지 않음
    E809 = (
        809,
        "cannot classify classes from input data",
    )  # classificaion 모델에서 분류가 불가능한 학습 데이터를 받음
    E810 = 810, "no training data and no target metric"  # 학습 데이터와 타겟 매트릭 두가지다 존재하지 않음
    E812 = 812, "wrong type of header"  # 헤더의 타입이 dictionary로 반환되지 않음
    E816 = (
        816,
        "'None' is not in list",
    )  # 학습을 할때 빈 시간에 None이 들어갔어야 됐는데 하나도 None이 들어가지 않음
    E817 = (
        817,
        "collected guid exceeds max tiers",
    )  # (이상거래코드탐지) 수집된 GUID가 max tiers보다 큼. 잘못 수집된 GUID
    E818 = (
        818,
        "no target metric data found on DB",
    )  # (이벤트예측)학습에 필요한 기준지표 데이터가 없음(빈 데이터프레임)
    E819 = (
        819,
        "specific xcode doesn't have enough training data",
    )  # (FastBaseline) 개별 트랜잭션/거래코드의 학습데이터가 다른 트랜잭션/거래코드보다 적음
    E820 = (
        820,
        "business day training mode mismatched",
    )  # (FastBaseline) 개별 트랜잭션/거래코드의 학습데이터가 다른 트랜잭션/거래코드보다 적음
    E821 = (821, "NaN is in training dataframe") # Nan값이 학습데이터에 존재함
    E822 = 822, "Dataframe index duplicated" # 학습 데이터의 인덱스 중복
    E823 = 823, "business day pf input tx code is not trained" # 학습된 거래코드가 비즈니스데이 학습 되지 않음
    E824 = 824, "business_list is not included in header" # 비즈니스데이 정보가 header에 포함되지 않음
    E825 = 825, "window serving data error" # 최초 서빙 시 필요한 window size에 해당하는 만큼 데이터가 들어오지 않음
    E826 = 826, "Log anomaly Detection Dbsln Band Training Error" # 로그 이상탐지 밴드 생성 오류

    E830 = 830, "Key value does not exist in dictionary"  # dictionary(meta) 안에 key 값(파라미터)이 존재하지 않음
    E831 = 831, "훈련 데이터와 요청받은 meta column 정보가 달라 (collective) 학습할 수 없음"  # gdn 관계성

    E832 = 832, "seqattn 알고리즘 훈련 중 에러 발생"
    E833 = 833, "GDN 알고리즘 훈련 중 에러 발생"

    E850 = 850, "model does not in directory"  # 해당디렉토리에 모델이 존재하지 않음
    E860 = 860, "there is no failure history in data"  # 분석 날짜 구간에 탐지된 장애가 존재하지 않음
    E861 = 861, "Error occured while deleting previous model in DB" # DB에 이전 모델 삭제 시 오류 발생 (모델 동기화)
    E862 = 862, "Error occured while inserting new model into DB" # 모델 파일 DB insert 시 오류 발생 (모델 동기화)
    E888 = 888, "Unknown training error "  # unknown training error
    E889 = 889, "Training Memory Error"  # 학습 메모리 관련 이슈
    E899 = 0


    # 알고리즘/서빙 에러 (900~999)
    E900 = 900, "unknown predict error" # unknown predict error
    E901 = 901, "training model does not exist"
    E904 = 904, "target model load not completed"
    E910 = 910, "DynamicBaseLine error"  # dbsln 알고리즘 오류 (이상탐지 기본 알고리즘 오류)

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: int, description: str = None):
        self._description_ = description

    def __int__(self):
        return self.value

    @property
    def desc(self):
        return self._description_
