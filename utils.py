import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000, item_features=None, bad_department = ['GROCERY']):

    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))

    min_price = data['price'].quantile(0.001)
    max_price = data['price'].quantile(0.999)

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    last_sale = data.groupby('item_id')['trans_time'].min().reset_index()

    old_sales = last_sale[last_sale['trans_time'] > 365]
    data = data[~data['item_id'].isin(old_sales)]

    # Уберем не интересные для рекоммендаций категории (department)
    if len(item_features) > 0:
        goods_by_department = item_features.groupby('item_id')['department'].unique().reset_index()

        for department in bad_department:
            unlucky_department = goods_by_department[goods_by_department['department'] == 'GROCERY']
            data = data[~data['item_id'].isin(unlucky_department)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    sort_by_rounded_price = \
        data.groupby('item_id')['price'].mean().reset_index().round(2)

    cheap_goods = sort_by_rounded_price[sort_by_rounded_price['price'] < min_price]
    data = data[~data['item_id'].isin(cheap_goods)]

    # Уберем слишком дорогие товарыs
    expensive_goods = sort_by_rounded_price[sort_by_rounded_price['price'] > max_price]
    data = data[~data['item_id'].isin(expensive_goods)]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    # ...
    return data


def postfilter_items(user_id, recommednations):
    pass


if __name__ == "__main__":
    print ('Utils is main, smile')