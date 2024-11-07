import dataclasses
import pandas as pd
import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod

from scipy.constants import degree

from DataLoader import Dataset
from collections import defaultdict, Counter
from surprise import KNNWithMeans, Reader
from surprise import Dataset as SurpriseDataset
from mlxtend.frequent_patterns import apriori, association_rules
import itertools
from sklearn.linear_model import LogisticRegression
import scipy
from sklearn.decomposition import NMF

@dataclasses.dataclass(frozen=True)
# 추천 시스템 예측 결과
class RecommendResult:
    # 테스트 데이터셋의 예측 평갓값. RMSE 평가
    rating: pd.DataFrame
    # 키는 사용자 ID, 값은 추천 아이템 ID 리스트. 순위 지표 평가.
    user2items: Dict[int, List[int]]

class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

# 1. 무작위 추천
class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 사용자 ID와 아이템 ID에 대해 0부터 시작하는 인덱스를 할당한다
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        # 사용자 x 아이템의 행렬에서 각 셀의 예측 평갓값은 0.5~5.0의 균등 난수로 한다
        pred_matrix = np.random.uniform(0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids)))

        # rmse 평가용으로 테스트 데이터에 나오는 사용자와 아이템의 예측 평갓값을 저장한다
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            # 테스트 데이터의 아이템 ID가 학습용으로 등장하지 않는 경우도 난수를 저장한다
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue
            # 테스트 데이터에 나타나는 사용자 ID와 아이템 ID의 인덱스를 얻어, 평갓값 행렬값을 얻는다
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # 순위 평가용 데이터 작성
        # 각 사용자에 대한 추천 영화는, 해당 사용자가 아직 평가하지 않은 영화 중에서 무작위로 10개 작품으로 한다
        # 키는 사용자 ID, 값은 추천 아이템의 ID 리스트
        pred_user2items = defaultdict(list)
        # 사용자가 이미 평가한 영화를 저장한다
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)

# 2. 통계 정보나 특정 규칙에 기반한 추천
class PopularityRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 평갓값의 임곗값
        minimum_num_rating = kwargs.get("minimum_num_rating", 1)

        # 각 아이템별 평균 평갓값을 계산하고, 그 평균 평갓값을 예측값으로 사용한다
        movie_rating_average = dataset.train.groupby("movie_id").agg({"rating": np.mean})
        # 테스트 데이터에 예측값을 저장한다. 테스트 데이터에만 존재하는 아이템의 예측 평갓값은 0으로 한다
        movie_rating_predict = dataset.test.merge(
            movie_rating_average, on="movie_id", how="left", suffixes=("_test", "_pred")
        ).fillna(0)

        # 각 사용자에 대한 추천 영화는 해당 사용자가 아직 평가하지 않은 영화 중에서 평균값이 높은 10개 작품으로 한다
        # 단, 평가 건수가 적으면 노이즈가 커지므로 minimum_num_rating건 이상 평가가 있는 영화로 한정한다
        pred_user2items = defaultdict(list)
        user_watched_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        movie_stats = dataset.train.groupby("movie_id").agg({"rating": [np.size, np.mean]})
        atleast_flg = movie_stats["rating"]["size"] >= minimum_num_rating
        movies_sorted_by_rating = (
            movie_stats[atleast_flg].sort_values(by=("rating", "mean"), ascending=False).index.tolist()
        )

        for user_id in dataset.train.user_id.unique():
            for movie_id in movies_sorted_by_rating:
                if movie_id not in user_watched_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)

# 3. 사용자-사용자 메모리 기반 협업 필터링
class UMCFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:

        # 피어슨 상관 계수
        def peason_coefficient(u: np.ndarray, v: np.ndarray) -> float:
            u_diff = u - np.mean(u)
            v_diff = v - np.mean(v)
            numerator = np.dot(u_diff, v_diff)
            denominator = np.sqrt(sum(u_diff ** 2)) * np.sqrt(sum(v_diff ** 2))
            if denominator == 0:
                return 0.0
            return numerator / denominator

        is_naive = kwargs.get("is_naive", True)

        # 평갓값을 사용자 x 영화 행렬로 변환
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))

        # 예측 대상 사용자와 영화 그룹
        movie_rating_predict = dataset.test.copy()
        pred_user2items = defaultdict(list)

        if is_naive:
            # 예측 대상 사용자 ID
            test_users = movie_rating_predict.user_id.unique()

            # 예측 대상 사용자(사용자 1)에 주목한다
            for user1_id in test_users:
                similar_users = []
                similarities = []
                avgs = []

                # 사용자 1과 평갓값 행렬 안의 다른 사용자(사용자 2)와의 유사도를 산출한다
                for user2_id in user_movie_matrix.index:
                    if user1_id == user2_id:
                        continue

                    # 사용자 1과 사용자 2의 평갓값 벡터
                    u_1 = user_movie_matrix.loc[user1_id, :].to_numpy()
                    u_2 = user_movie_matrix.loc[user2_id, :].to_numpy()

                    # `u_1`과 `u_2` 모두에서 결손값이 없는 요소만 추출한 벡터를 얻는다
                    common_items = ~np.isnan(u_1) & ~np.isnan(u_2)

                    # 공통으로 평가한 아이템이 없는 경우는 스킵한다
                    if not common_items.any():
                        continue

                    u_1, u_2 = u_1[common_items], u_2[common_items]

                    # 피어슨 상관 계수를 사용해 사용자 1과 사용자 2의 유사도를 산출한다
                    rho_12 = peason_coefficient(u_1, u_2)

                    # 사용자 1과의 유사도가 0보다 큰 경우, 사용자 2를 유사 사용자로 간주한다
                    if rho_12 > 0:
                        similar_users.append(user2_id)
                        similarities.append(rho_12)
                        avgs.append(np.mean(u_2))

                # 사용자 1의 평균 평갓값
                avg_1 = np.mean(user_movie_matrix.loc[user1_id, :].dropna().to_numpy())

                # 예측 대상의 영화 ID
                test_movies = movie_rating_predict[movie_rating_predict["user_id"] == user1_id].movie_id.values
                # 예측할 수 없는 영화에 대한 평갓값은 사용자 1의 평균 평갓값으로 한다
                movie_rating_predict.loc[(movie_rating_predict["user_id"] == user1_id), "rating_pred"] = avg_1

                if similar_users:
                    for movie_id in test_movies:
                        if movie_id in movie_id2index:
                            r_xy = user_movie_matrix.loc[similar_users, movie_id].to_numpy()
                            rating_exists = ~np.isnan(r_xy)

                            # 유사 사용자가 대상이 되는 영화에 대한 평갓값을 갖지 않는 경우는 스킵한다
                            if not rating_exists.any():
                                continue

                            r_xy = r_xy[rating_exists]
                            rho_1x = np.array(similarities)[rating_exists]
                            avg_x = np.array(avgs)[rating_exists]
                            r_hat_1y = avg_1 + np.dot(rho_1x, (r_xy - avg_x)) / rho_1x.sum()

                            # 예측 평갓값을 저장한다
                            movie_rating_predict.loc[
                                (movie_rating_predict["user_id"] == user1_id)
                                & (movie_rating_predict["movie_id"] == movie_id),
                                "rating_pred",
                            ] = r_hat_1y


        else:
            # Surprise용으로 데이터를 가공한다
            reader = Reader(rating_scale=(0.5, 5))
            data_train = SurpriseDataset.load_from_df(
                dataset.train[["user_id", "movie_id", "rating"]], reader
            ).build_full_trainset()

            sim_options = {"name": "pearson", "user_based": True}  # 유사도를 계산하는 방법을 지정한다  # False로 하면 아이템 기반이 된다
            knn = KNNWithMeans(k=30, min_k=1, sim_options=sim_options)
            knn.fit(data_train)

            # 학습 데이터셋에서 평갓값이 없는 사용자와 아이템의 조합을 준비
            data_test = data_train.build_anti_testset(None)
            predictions = knn.test(data_test)

            def get_top_n(predictions, n=10):
                # 각 사용자별로 예측된 아이템을 저장한다
                top_n = defaultdict(list)
                for uid, iid, true_r, est, _ in predictions:
                    top_n[uid].append((iid, est))

                # 상요자별로 아이템을 예측 평갓값순으로 나열하고 상위 n개를 저장한다
                for uid, user_ratings in top_n.items():
                    user_ratings.sort(key=lambda x: x[1], reverse=True)
                    top_n[uid] = [d[0] for d in user_ratings[:n]]

                return top_n

            pred_user2items = get_top_n(predictions, n=10)

            average_score = dataset.train.rating.mean()
            pred_results = []
            for _, row in dataset.test.iterrows():
                user_id = row["user_id"]
                movie_id = row["movie_id"]
                # 학습 데이터에 존재하지 않고 테스트 데이터에만 존재하는 사용자나 영화에 관한 예측 평갓값은 전체 평균 평갓값으로 한다
                if user_id not in user_id2index or movie_id not in movie_id2index:
                    pred_results.append(average_score)
                    continue
                pred_score = knn.predict(uid=user_id, iid=movie_id).est
                pred_results.append(pred_score)
            movie_rating_predict["rating_pred"] = pred_results

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)

# 4. 연관 규칙
class AssociationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 평갓값의 임곗값
        min_support = kwargs.get("min_support", 0.06)
        min_threshold = kwargs.get("min_threshold", 1)

        # 사용자 x 영화 행렬 형식으로 변경
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")

        # 라이브러리 사용을 위해 4 이상의 평갓값은 1, 4 미만의 평갓값은 0으로 한다
        user_movie_matrix[user_movie_matrix < 4] = 0
        user_movie_matrix[user_movie_matrix.isnull()] = 0
        user_movie_matrix[user_movie_matrix >= 4] = 1

        # 지지도가 높은 영화
        freq_movies = apriori(user_movie_matrix, min_support=min_support, use_colnames=True)
        # 어소시에이션 규칙 계산(리프트값이 높은 순으로 표시)
        rules = association_rules(freq_movies, metric="lift", min_threshold=min_threshold)

        # 어소시에이션 규칙을 사용해, 각 사용자가 아직 평가하지 않은 영화를 10개 추천한다
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        # 학습용 데이터에서 평갓값이 4 이상인 것만 얻는다
        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]

        for user_id, data in movielens_train_high_rating.groupby("user_id"):
            # 사용자가 직전에 평가한 5개의 영화를 얻는다
            input_data = data.sort_values("timestamp")["movie_id"].tolist()[-5:]
            # 그 영화들이 조건부에 하나라도 포함되는 어소시에이션 규칙을 검출한다
            matched_flags = rules.antecedents.apply(lambda x: len(set(input_data) & x)) >= 1

            # 어소시에이션 규칙의 귀결부의 영화를 리스트에 저장하고, 등록 빈도 수로 정렬해 사용자가 아직 평가하지 않았다면, 추천 목록에 추가한다
            consequent_movies = []
            for i, row in rules[matched_flags].sort_values("lift", ascending=False).iterrows():
                consequent_movies.extend(row["consequents"])
            # 등록 빈도 세기
            counter = Counter(consequent_movies)
            for movie_id, movie_cnt in counter.most_common():
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                # 추천 리스트가 10이 되면 종료한다
                if len(pred_user2items[user_id]) == 10:
                    break

        # 어소시에이션 규칙에서는 평갓값을 예측하지 않으므로, rmse 평가는 수행하지 않는다(편의상, 테스트 데이터의 예측값을 그대로 반환).
        return RecommendResult(dataset.test.rating, pred_user2items)

# 5. Recommender Regression
class LogisticRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        threshold = 3.0  # Threshold for binary classification (example: rating >= 3.0 is considered "good")
        dataset.train['rating_binary'] = dataset.train['rating'].apply(lambda x: 1 if x >= threshold else 0)

        # 사용자 x 영화 평점 행렬을 이진 레이블로 변환합니다.
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating_binary")
        #user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        #movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))

        # 평갓값을 사용자 x 영화의 행렬로 변환한다. 결손값은 평균값 또는 0으로 채운다
        #user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating_binary")
        #user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        #movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))

        # 학습에 사용하는 학습용 데이터 중 사용자와 영화의 조합을 얻는다
        train_keys = dataset.train[["user_id", "movie_id"]]
        train_y = dataset.train['rating_binary'].values

        # 평갓값을 예측하고자 하는 테스트용 데이터 안의 사용자와 영화의 조합을 얻는다
        test_keys = dataset.test[["user_id", "movie_id"]]
        # 순위 형식의 추천 리스르 작성을 윟 학습용 데이터에 존재하는 모든 사용자와 모든 영화의 조합을 저장한다
        train_all_keys = user_movie_matrix.stack(dropna=False).reset_index()[["user_id", "movie_id"]]

        # 특징량을 작성한다
        train_x = train_keys.copy()
        test_x = test_keys.copy()
        train_all_x = train_all_keys.copy()

        # 학습용 데이터에 존재하는 사용자별 평갓값의 최솟값, 최댓값, 평균값
        # 및, 영화별 평갓값의 최솟값, 최댓값, 평균값을 특징량ㅇ로 추가한다
        aggregators = ["min", "max", "mean"]
        user_features = dataset.train.groupby("user_id").rating.agg(aggregators).to_dict()
        movie_features = dataset.train.groupby("movie_id").rating.agg(aggregators).to_dict()
        for agg in aggregators:
            train_x[f"u_{agg}"] = train_x["user_id"].map(user_features[agg])
            test_x[f"u_{agg}"] = test_x["user_id"].map(user_features[agg])
            train_all_x[f"u_{agg}"] = train_all_x["user_id"].map(user_features[agg])
            train_x[f"m_{agg}"] = train_x["movie_id"].map(movie_features[agg])
            test_x[f"m_{agg}"] = test_x["movie_id"].map(movie_features[agg])
            train_all_x[f"m_{agg}"] = train_all_x["movie_id"].map(movie_features[agg])
        # 테스트용 데이터에만 존재하는 사용자나 영화의 특징량을, 학습용 데이터 전체의 평균 평갓값으로 채운다
        average_rating = train_y.mean()
        test_x.fillna(average_rating, inplace=True)

        # 영화가 특정한 genre에 있는지를 나타태는 특징량을 추가
        movie_genres = dataset.item_content[["movie_id", "genre"]]
        genres = set(list(itertools.chain(*movie_genres.genre)))
        for genre in genres:
            movie_genres[f"is_{genre}"] = movie_genres.genre.apply(lambda x: genre in x)
        movie_genres.drop("genre", axis=1, inplace=True)
        train_x = train_x.merge(movie_genres, on="movie_id")
        test_x = test_x.merge(movie_genres, on="movie_id")
        train_all_x = train_all_x.merge(movie_genres, on="movie_id")

        # 특징량으로서는 사용하지 않는 정보를 삭제
        train_x = train_x.drop(columns=["user_id", "movie_id"])
        test_x = test_x.drop(columns=["user_id", "movie_id"])
        train_all_x = train_all_x.drop(columns=["user_id", "movie_id"])

        # 6. Logistic Regression 모델을 학습
        reg = LogisticRegression(max_iter=10000, n_jobs=-1)  # 반복 횟수를 늘려서 수렴 가능성 증가
        reg.fit(train_x.values, train_y)

        # 테스트용 데이터 안의 사용자와 영화의 조합에 대한 추천 여부를 예측한다
        test_pred = reg.predict(test_x.values)

        movie_rating_predict = test_keys.copy()
        movie_rating_predict["rating_pred"] = test_pred

        # 학습용 데이터에 존재하는 모든 사용자와 모든 영화의 조합애 대해 평갓값을 예측한다
        train_all_pred = reg.predict(train_all_x.values)

        pred_train_all = train_all_keys.copy()
        pred_train_all["rating_pred"] = train_all_pred
        pred_matrix = pred_train_all.pivot(index="user_id", columns="movie_id", values="rating_pred")

        # 사용자가 학습용 데이터 안에서 평가하지 않은 영화 중에서
        # 예측 평갓값이 높은 순으로 10건의 영화를 순위 형식의 추천 리스트로 한다
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in dataset.train.user_id.unique():
            movie_indexes = np.argsort(-pred_matrix.loc[user_id, :]).values
            for movie_index in movie_indexes:
                movie_id = user_movie_matrix.columns[movie_index]
                if movie_id not in (user_evaluated_movies[user_id]):
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)

# 6. SVD
class SVDRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 결손값을 채우는 방법
        fillna_with_zero = kwargs.get("fillna_with_zero", False)
        factors = kwargs.get("factors", 5)

        # 평갓값을 사용자 x 영화의 행렬로 변환한다. 평갓값 또는 0으로 채운다.
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))
        if fillna_with_zero:
            matrix = user_movie_matrix.fillna(0).to_numpy()
        else:
            matrix = user_movie_matrix.fillna(dataset.train.rating.mean()).to_numpy()

        # 인자 수 k로 특이값 분해를 수행한다
        P, S, Qt = scipy.sparse.linalg.svds(matrix, k=factors)

        # 예측 평갓값 행렬
        pred_matrix = np.dot(np.dot(P, np.diag(S)), Qt)

        # 학습용에 나오지 않는 사용자나 영화의 예측 평갓값은 평균 평갓값으로 한다
        average_score = dataset.train.rating.mean()
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            if user_id not in user_id2index or row["movie_id"] not in movie_id2index:
                pred_results.append(average_score)
                continue
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # 각 사용자에 대한 추천 영화는 해당 사용자가 아직 평가하지 않은 영화중에서 예측값이 높은 순으로 한다
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in dataset.train.user_id.unique():
            if user_id not in user_id2index:
                continue
            user_index = user_id2index[row["user_id"]]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = user_movie_matrix.columns[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)

# 7. NMF
class NMFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 결손값을 채우는 방법
        fillna_with_zero = kwargs.get("fillna_with_zero", False)
        factors = kwargs.get("factors", 5)

        # 평갓값을 사용자 x 영화의 행렬로 변환한다. 결손값은 평균값 또는 0으로 채운다)
        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")
        user_id2index = dict(zip(user_movie_matrix.index, range(len(user_movie_matrix.index))))
        movie_id2index = dict(zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns))))
        if fillna_with_zero:
            matrix = user_movie_matrix.fillna(0).to_numpy()
        else:
            matrix = user_movie_matrix.fillna(dataset.train.rating.mean()).to_numpy()

        nmf = NMF(n_components=factors)
        nmf.fit(matrix)
        P = nmf.fit_transform(matrix)
        Q = nmf.components_

        # 예측 평갓값 행렬
        pred_matrix = np.dot(P, Q)

        # 학습용에 나오지 않은 사용자나 영화의 예측 평갓값은 평균 평갓값으로 한다
        average_score = dataset.train.rating.mean()
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            if user_id not in user_id2index or row["movie_id"] not in movie_id2index:
                pred_results.append(average_score)
                continue
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # 각 사용자에게 대한 추천 영화는 그 사용자가 아직 평가하지 않은 영화 중에서 예측값이 높은 순으로 한다
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in dataset.train.user_id.unique():
            if user_id not in user_id2index:
                continue
            user_index = user_id2index[row["user_id"]]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = user_movie_matrix.columns[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break

        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)
