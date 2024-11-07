from DataLoader import DataLoader
from Recommender import RandomRecommender, PopularityRecommender, AssociationRecommender, UMCFRecommender, LogisticRecommender
# from Recommender import SVDRecommender, NMFRecommender, MFRecommender, IMFRecommender, BPRRecommender, FMRecommender
# from Recommender import LDAContentRecommender, LDACollaborationRecommender, Word2vecRecommender, Item2vecRecommender
from MetricCalculator import MetricCalculator

path = './ml-10M100K'

# 1. MovieLens 데이터 로딩
data_loader = DataLoader(num_users=500, num_test_items=5, data_path=path)
movielens = data_loader.load()

# 2. 추천 알고리즘 구현
recommender = LogisticRecommender()
recommend_result = recommender.recommend(movielens)

# 3. 평가 지표 계산
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(movielens.test.rating.tolist(), recommend_result.rating.tolist(),
                                 movielens.test_user2items, recommend_result.user2items, k=10)
print(metrics)
