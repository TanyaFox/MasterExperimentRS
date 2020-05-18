from collections import defaultdict
import pandas as pd
import numpy as np
import random
from scipy import sparse
from scipy import stats
from sklearn.preprocessing import MultiLabelBinarizer
import tensorrec


# Оценка метрик качества рекомендаций (для RMSE в качестве графа потерь)
def check_results(ranks, train, test):
    train_precision_at_10 = tensorrec.eval.precision_at_k(
        test_interactions=train,
        predicted_ranks=ranks,
        k=10
    ).mean()
    test_precision_at_10 = tensorrec.eval.precision_at_k(
        test_interactions=test,
        predicted_ranks=ranks,
        k=10
    ).mean()

    train_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=train,
        predicted_ranks=ranks,
        k=10
    ).mean()
    test_recall_at_10 = tensorrec.eval.recall_at_k(
        test_interactions=test,
        predicted_ranks=ranks,
        k=10
    ).mean()

    train_f_at_10 = tensorrec.eval.f1_score_at_k(
        predicted_ranks=ranks,
        test_interactions=train,
        k=10
    ).mean()
    test_f_at_10 = tensorrec.eval.f1_score_at_k(
        predicted_ranks=ranks,
        test_interactions=test,
        k=10
    ).mean()

    print("Precision at 10: \n Train: {:.4f} Test: {:.4f}".format(train_precision_at_10,
                                                               test_precision_at_10))

    print("Recall at 10: \n Train: {:.4f} Test: {:.4f}".format(train_recall_at_10,
                                                            test_recall_at_10))

    print("F at 10: \n Train: {:.4f} Test: {:.4f}".format(train_f_at_10,
                                                               test_f_at_10))


# Оценка значимости в предсказаниях
def test_significance(y1, y2):
    # Тестируем гипотезу на нормальность
    y1_shapiro = stats.shapiro(y1)
    print(y1_shapiro)
    y2_shapiro = stats.shapiro(y2)
    print(y2_shapiro)

    if y1_shapiro[1] >= 0.05 and y2_shapiro[1] >= 0.05:
        print('Distributions of quantities are normal')
        # Тестируем гипотезу на равенство дисперсий
        fligner_test = stats.fligner(y1, y2)
        print(fligner_test)

        # Т-тест (только если нормальное распределение)
        if fligner_test[1] < 0.05:
            print('Variances are not equal')
            ttest_result = stats.ttest_ind(y1, y2, equal_var=False)
        else:
            print('Variances are equal')
            ttest_result = stats.ttest_ind(y1, y2, equal_var=True)

        print(ttest_result)
        if ttest_result[1] >= 0.05:
            print('Differences in predictions are not significant.')
        else:
            print('Differences in predictions are significant.')
    else:
        print('Distributions of quantities are not normal')
        # Тест Вилкоксона (если распределение не подчиняется нормальному закону)
        wilcoxon_result = stats.wilcoxon(y1, y2)
        print(wilcoxon_result)
        if wilcoxon_result[1] >= 0.05:
            print('Differences in predictions are not significant.')
        else:
            print('Differences in predictions are significant.')


# Читаем данные
ratings = pd.read_csv('DATA/RATINGS.csv', sep=';')
songs = pd.read_csv('DATA/SONGS.csv', sep=';')
users = pd.read_csv('DATA/USER_STATES.csv', sep=';')


# Приводим датафреймы оценок с списку списков и удаляем временную метку
ratings = ratings.drop(['timestamp'], axis=1)
list_of_ratings = []
list_of_songs = []
list_of_users = []

for row in ratings.values:
    list_of_ratings.append(list(row))

for row in songs.values:
    list_of_songs.append(list(row))

for row in users.values:
    list_of_users.append(list(row))


# Переразмечаем айдишники для внутреннего использования
data_to_internal_user_ids = defaultdict(lambda: len(data_to_internal_user_ids))
data_to_internal_item_ids = defaultdict(lambda: len(data_to_internal_item_ids))
for row in list_of_ratings:
    row[0] = data_to_internal_user_ids[int(row[0])]
    row[1] = data_to_internal_item_ids[int(row[1])]
    row[2] = int(row[2])
n_users = len(data_to_internal_user_ids)
n_items = len(data_to_internal_item_ids)


# Строим матрицу scipy sparse для оценок
def interactions_list_to_sparse_matrix(interactions):
    users_column, items_column, ratings_column = zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)),
                             shape=(n_users, n_items))


# Перемешиваем датасет для случайного отбора и делим на 70%/30% (обучающая/тестовая)
random.shuffle(list_of_ratings)
cutoff = int(.7 * len(list_of_ratings))
train_ratings = list_of_ratings[:cutoff]
test_ratings = list_of_ratings[cutoff:]
print("{} train ratings, {} test ratings".format(len(train_ratings), len(test_ratings)))


# Строим матрицы с помощью метода, созданного ранее - для оценок и признаков
sparse_train_ratings = interactions_list_to_sparse_matrix(train_ratings)
sparse_test_ratings = interactions_list_to_sparse_matrix(test_ratings)
user_indicator_features = sparse.identity(n_users)
item_indicator_features = sparse.identity(n_items)


# Соотносим ID песен с внутренними ID песен и отслеживаем фичи
songs_artists_by_internal_id = {}
songs_names_by_internal_id = {}
songs_danceability_by_internal_id = {}
songs_energy_by_internal_id = {}
songs_loudness_by_internal_id = {}
songs_mode_by_internal_id = {}
songs_speechness_by_internal_id = {}
songs_acousticness_by_internal_id = {}
songs_instrumentalness_by_internal_id = {}
songs_liveness_by_internal_id = {}
songs_tempo_by_internal_id = {}
songs_duration_by_internal_id = {}
songs_genre_by_internal_id = {}

for row in list_of_songs:
    row[0] = data_to_internal_item_ids[int(row[0])]  # Map to IDs
    row[2] = row[2].replace(' ', '')
    songs_artists_by_internal_id[row[0]] = row[1]
    songs_names_by_internal_id[row[0]] = row[2]
    songs_danceability_by_internal_id[row[0]] = row[3]
    songs_energy_by_internal_id[row[0]] = row[4]
    songs_loudness_by_internal_id[row[0]] = row[5]
    songs_mode_by_internal_id[row[0]] = row[6]
    songs_speechness_by_internal_id[row[0]] = row[7]
    songs_acousticness_by_internal_id[row[0]] = row[8]
    songs_instrumentalness_by_internal_id[row[0]] = row[9]
    songs_liveness_by_internal_id[row[0]] = row[10]
    songs_tempo_by_internal_id[row[0]] = row[11]
    songs_duration_by_internal_id[row[0]] = row[12]
    songs_genre_by_internal_id[row[0]] = row[13]

# Списки фичей для песен
songs_artists = [songs_artists_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_names = [songs_names_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_danceability = [songs_danceability_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_energy = [songs_energy_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_loudness = [songs_loudness_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_mode = [songs_mode_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_speechness = [songs_speechness_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_acousticness = [songs_acousticness_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_instrumentalness = [songs_instrumentalness_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_liveness = [songs_liveness_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_tempo = [songs_tempo_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_duration = [songs_duration_by_internal_id[internal_id] for internal_id in range(n_items)]
songs_genre = [songs_genre_by_internal_id[internal_id] for internal_id in range(n_items)]


# Соотносим ID пользователей с внутренними ID и отслеживаем фичи
user_timestamp_by_internal_id = {}
user_gender_by_internal_id = {}
user_age_by_internal_id = {}
user_is_active_by_internal_id = {}
user_mood_by_internal_id = {}
user_reaction_by_internal_id = {}
user_temperament_by_internal_id = {}

for row in list_of_users:
    row[0] = data_to_internal_user_ids[int(row[0])] # Map to IDs
    user_timestamp_by_internal_id[row[0]] = row[1]
    user_gender_by_internal_id[row[0]] = row[2]
    user_age_by_internal_id[row[0]] = row[3]
    user_is_active_by_internal_id[row[0]] = row[4]
    user_mood_by_internal_id[row[0]] = row[5]
    user_reaction_by_internal_id[row[0]] = row[6]
    user_temperament_by_internal_id[row[0]] = row[7]

# Списки фичей пользователей
user_timestamp = [user_timestamp_by_internal_id[internal_id] for internal_id in range(n_users)]
user_gender = [user_gender_by_internal_id[internal_id] for internal_id in range(n_users)]
user_age = [user_age_by_internal_id[internal_id] for internal_id in range(n_users)]
user_is_active = [user_is_active_by_internal_id[internal_id] for internal_id in range(n_users)]
user_mood = [user_mood_by_internal_id[internal_id] for internal_id in range(n_users)]
user_reaction = [user_reaction_by_internal_id[internal_id] for internal_id in range(n_users)]
user_temperament = [user_temperament_by_internal_id[internal_id] for internal_id in range(n_users)]


# Бинаризуем фичи с помощью scikit's MultiLabelBinarizer
songs_artists_features = MultiLabelBinarizer().fit_transform(songs_artists)
songs_names_features = MultiLabelBinarizer().fit_transform(songs_names)
songs_genre_features = MultiLabelBinarizer().fit_transform(songs_genre)

user_mood_features = MultiLabelBinarizer().fit_transform(user_mood)
user_temperament_features = MultiLabelBinarizer().fit_transform(user_temperament)


# Приведение фичей к sparse matrix, которая нужна на вход для TensorRec
songs_artists_features = sparse.coo_matrix(songs_artists_features)
songs_names_features = sparse.coo_matrix(songs_names_features)
songs_danceability_features = sparse.coo_matrix(songs_danceability)
songs_energy_features = sparse.coo_matrix(songs_energy)
songs_loudness_features = sparse.coo_matrix(songs_loudness)
songs_mode_features = sparse.coo_matrix(songs_mode)
songs_speechness_features = sparse.coo_matrix(songs_speechness)
songs_acousticness_features = sparse.coo_matrix(songs_acousticness)
songs_instrumentalness_features = sparse.coo_matrix(songs_instrumentalness)
songs_liveness_features = sparse.coo_matrix(songs_liveness)
songs_tempo_features = sparse.coo_matrix(songs_tempo)
songs_duration_features = sparse.coo_matrix(songs_duration)
songs_genre_features = sparse.coo_matrix(songs_genre_features)

user_timestamp_features = sparse.coo_matrix(user_timestamp)
user_gender_features = sparse.coo_matrix(user_gender)
user_age_features = sparse.coo_matrix(user_age)
user_is_active_features = sparse.coo_matrix(user_is_active)
user_mood_features = sparse.coo_matrix(user_mood_features)
user_reaction_features = sparse.coo_matrix(user_reaction)
user_temperament_features = sparse.coo_matrix(user_temperament_features)


# Слияние фич в наборы для гибридной рекомендательной системы и коллаборативной фильтрации
full_item_features = sparse.hstack([item_indicator_features, songs_artists_features, songs_names_features,
                                    np.reshape(songs_danceability_features, (songs_danceability_features.shape[1],1)),
                                    np.reshape(songs_energy_features, (songs_energy_features.shape[1],1)),
                                    np.reshape(songs_loudness_features, (songs_loudness_features.shape[1],1)),
                                    np.reshape(songs_mode_features, (songs_mode_features.shape[1],1)),
                                    np.reshape(songs_speechness_features, (songs_speechness_features.shape[1],1)),
                                    np.reshape(songs_acousticness_features, (songs_acousticness_features.shape[1],1)),
                                    np.reshape(songs_instrumentalness_features, (songs_instrumentalness_features.shape[1],1)),
                                    np.reshape(songs_liveness_features, (songs_liveness_features.shape[1],1)),
                                    np.reshape(songs_tempo_features, (songs_tempo_features.shape[1],1)),
                                    np.reshape(songs_duration_features, (songs_duration_features.shape[1],1)),
                                    songs_genre_features])

cut_user_features = sparse.hstack([user_indicator_features,
                                   np.reshape(user_gender_features, (user_gender_features.shape[1], 1)),
                                   np.reshape(user_age_features, (user_age_features.shape[1], 1))])

full_user_features = sparse.hstack([user_indicator_features, user_mood_features, user_temperament_features,
                                    np.reshape(user_age_features, (user_age_features.shape[1],1)),
                                    np.reshape(user_timestamp_features, (user_timestamp_features.shape[1],1)),
                                    np.reshape(user_gender_features, (user_gender_features.shape[1],1)),
                                    np.reshape(user_is_active_features, (user_is_active_features.shape[1],1)),
                                    np.reshape(user_reaction_features, (user_reaction_features.shape[1],1))])


# Коллаборативная фильтрация
print("RMSE matrix factorization collaborative filter (cut):")
ranking_cf_model = tensorrec.TensorRec(n_components=5)
ranking_cf_model.fit(interactions=sparse_train_ratings,
                     user_features=cut_user_features,
                     item_features=item_indicator_features)
cut_cf_predicted_ranks = ranking_cf_model.predict_rank(user_features=cut_user_features,
                                                       item_features=item_indicator_features)
check_results(cut_cf_predicted_ranks, sparse_train_ratings, sparse_test_ratings)

print("RMSE matrix factorization collaborative filter (full):")
ranking_cf_full_model = tensorrec.TensorRec(n_components=5)
ranking_cf_full_model.fit(interactions=sparse_train_ratings,
                          user_features=full_user_features,
                          item_features=item_indicator_features)
predicted_ranks = ranking_cf_full_model.predict_rank(user_features=full_user_features,
                                                     item_features=item_indicator_features)
check_results(predicted_ranks, sparse_train_ratings, sparse_test_ratings)


# Гибридная модель
print("Hybrid recommender (cut features):")
cut_hybrid_model = tensorrec.TensorRec(n_components=5)
cut_hybrid_model.fit(interactions=sparse_train_ratings,
                     user_features=cut_user_features,
                     item_features=full_item_features)
cut_predicted_ranks = cut_hybrid_model.predict_rank(user_features=cut_user_features,
                                            item_features=full_item_features)
check_results(cut_predicted_ranks, sparse_train_ratings, sparse_test_ratings)

print("Hybrid recommender (full features):")
full_hybrid_model = tensorrec.TensorRec(n_components=5)
full_hybrid_model.fit(interactions=sparse_train_ratings,
                      user_features=full_user_features,
                      item_features=full_item_features)
full_predicted_ranks = full_hybrid_model.predict_rank(user_features=full_user_features,
                                                      item_features=full_item_features)
check_results(full_predicted_ranks, sparse_train_ratings, sparse_test_ratings)


# Оценим значимость в предсказании модели на примере 0-ого пользователя
# Убираем признаки 0 пользователя из матрицы признаков и предсказываем набор песен только для 0 пользователя
user0_features_cut = sparse.csr_matrix(cut_user_features)[0]
user0_features_full = sparse.csr_matrix(full_user_features)[0]

print('CF Evaluation:')
user0_rankings_cf = ranking_cf_model.predict_rank(user_features=user0_features_cut,
                                                  item_features=item_indicator_features)[0]
user0_predictions_cf = ranking_cf_model.predict(user_features=user0_features_cut,
                                                item_features=item_indicator_features)[0]

user0_rankings_cf_full = ranking_cf_full_model.predict_rank(user_features=user0_features_full,
                                          item_features=item_indicator_features)[0]
user0_predictions_cf_full = ranking_cf_full_model.predict(user_features=user0_features_full,
                                             item_features=item_indicator_features)[0]
# Тестируем значимость
test_significance(user0_predictions_cf, user0_predictions_cf_full)


print('Hybrid:')
user0_rankings = cut_hybrid_model.predict_rank(user_features=user0_features_cut,
                                               item_features=full_item_features)[0]
user0_predictions = cut_hybrid_model.predict(user_features=user0_features_cut,
                                             item_features=full_item_features)[0]
user0_rankings_full = full_hybrid_model.predict_rank(user_features=user0_features_full,
                                                     item_features=full_item_features)[0]
user0_predictions_full = full_hybrid_model.predict(user_features=user0_features_full,
                                                   item_features=full_item_features)[0]
# Тестируем значимость
test_significance(user0_predictions, user0_predictions_full)


# Попытка улучшить качество модели за счет применения графа потерь WMRB (weighted margin-rank batch)
# (так как в исходных данных оценки бинарны, на качество практически не влияет в итоге)
# Подготовка датасетов train/test с оценками, которые выше >= 0.6
sparse_train_ratings_06plus = sparse_train_ratings.multiply(sparse_train_ratings >= 0.6)
sparse_test_ratings_06plus = sparse_test_ratings.multiply(sparse_test_ratings >= 0.6)


# Коллаборативная фильтрация
print("WMRB matrix factorization collaborative filter (cut):")
ranking_cf_model = tensorrec.TensorRec(n_components=5,
                                       loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
ranking_cf_model.fit(interactions=sparse_train_ratings_06plus,
                     user_features=cut_user_features,
                     item_features=item_indicator_features,
                     n_sampled_items=int(n_items * .02))
cut_cf_predicted_ranks = ranking_cf_model.predict_rank(user_features=cut_user_features,
                                                       item_features=item_indicator_features)
check_results(cut_cf_predicted_ranks, sparse_train_ratings_06plus, sparse_test_ratings_06plus)

print("WMRB matrix factorization collaborative filter (full):")
ranking_cf_full_model = tensorrec.TensorRec(n_components=5,
                                            loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
ranking_cf_full_model.fit(interactions=sparse_train_ratings_06plus,
                          user_features=full_user_features,
                          item_features=item_indicator_features,
                          n_sampled_items=int(n_items * .02))
predicted_ranks = ranking_cf_full_model.predict_rank(user_features=full_user_features,
                                                     item_features=item_indicator_features)

check_results(predicted_ranks, sparse_train_ratings_06plus, sparse_test_ratings_06plus)


# Гибридная модель
print("Hybrid recommender (cut features):")
cut_hybrid_model = tensorrec.TensorRec(
    n_components=5,
    loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
cut_hybrid_model.fit(interactions=sparse_train_ratings_06plus,
                     user_features=cut_user_features,
                     item_features=full_item_features,
                     n_sampled_items=int(n_items * .02))
cut_predicted_ranks = cut_hybrid_model.predict_rank(user_features=cut_user_features,
                                                    item_features=full_item_features)
check_results(cut_predicted_ranks, sparse_train_ratings_06plus, sparse_test_ratings_06plus)


print("Hybrid recommender (full features):")
full_hybrid_model = tensorrec.TensorRec(
    n_components=5,
    loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
full_hybrid_model.fit(interactions=sparse_train_ratings_06plus,
                      user_features=full_user_features,
                      item_features=full_item_features,
                      n_sampled_items=int(n_items * .02))
full_predicted_ranks = full_hybrid_model.predict_rank(user_features=full_user_features,
                                                      item_features=full_item_features)
check_results(full_predicted_ranks, sparse_train_ratings_06plus, sparse_test_ratings_06plus)

# Также оценим значимость различия в предсказаниях на примере 0-ого пользователя
# Предсказываем набор песен только для 0 пользователя
print('WMRB CF:')
user0_rankings_cf = ranking_cf_model.predict_rank(user_features=user0_features_cut,
                                                  item_features=item_indicator_features)[0]
user0_predictions_cf = ranking_cf_model.predict(user_features=user0_features_cut,
                                                item_features=item_indicator_features)[0]

user0_rankings_cf_full = ranking_cf_full_model.predict_rank(user_features=user0_features_full,
                                                            item_features=item_indicator_features)[0]
user0_predictions_cf_full = ranking_cf_full_model.predict(user_features=user0_features_full,
                                                          item_features=item_indicator_features)[0]
# Тестируем значимость
test_significance(user0_predictions_cf, user0_predictions_cf_full)


print('WMRB Hybrid:')
user0_rankings = cut_hybrid_model.predict_rank(user_features=user0_features_cut,
                                               item_features=full_item_features)[0]
user0_predictions = cut_hybrid_model.predict(user_features=user0_features_cut,
                                             item_features=full_item_features)[0]

user0_rankings_full = full_hybrid_model.predict_rank(user_features=user0_features_full,
                                                     item_features=full_item_features)[0]
user0_predictions_full = full_hybrid_model.predict(user_features=user0_features_full,
                                                   item_features=full_item_features)[0]
# Тестируем значимость
test_significance(user0_predictions, user0_predictions_full)