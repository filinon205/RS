import pandas as pd
import numpy as np

"""
Metrics
"""

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1


def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)


def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)[:k]
    recommended_list = np.concatenate([recommended_list[:k], np.zeros(len(recommended_list) - k)])
    flags = np.isin(recommended_list, bought_list)
    precision = np.dot(flags, prices_recommended) / prices_recommended.sum()
    return precision


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[:k]
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)

    # better solution
    extended_recommended_list = np.zeros(len(recommended_list))
    extended_recommended_list[:k] = recommended_list[:k]

    # worse solution
    #     recommended_list = np.concatenate([recommended_list[:k],np.zeros(len(recommended_list)-k)])

    flags = np.isin(extended_recommended_list, bought_list)
    recall = np.dot(flags, prices_recommended) / prices_bought.sum()

    return recall


# AP@K - average precision at k
def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(recommended_list, bought_list)
    if sum(flags) == 0:
        return 0
    sum_ = 0
    for i in range(1, k + 1):
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
    result = sum_ / k
    return result


# MAP@k (Mean Average Precision@k)
def map_k(recommended_list, bought_list, k=5):
    """
    на вход должны подаваться два списка списков (lists of lists) покупок и рекомендаций.
    """
    return np.mean([ap_k(a,p,k) for a,p in zip(bought_list, recommended_list)])

# Mean Reciprocal Rank
def reciprocal_rank(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    result = np.mean(1 / (flags.nonzero()[0][0] + 1)) if flags.size else 0
    return result

if __name__ == "__main__":
    print ('Metrics is main, smile')