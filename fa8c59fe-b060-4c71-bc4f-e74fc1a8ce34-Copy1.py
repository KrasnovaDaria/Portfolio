#!/usr/bin/env python
# coding: utf-8

# # Procrastinate Pro+

# Предоставлены данные о пользователях, привлечённых с 1 мая по 27 октября 2019 года:
# - лог сервера с данными об их посещениях,
# - выгрузка их покупок за этот период,
# - рекламные расходы.
# 
# Цель исследования:
# 
# - Выделить причины неэффективности привлечения пользователей
# - ВЫявить точки роста
# - Сформулировать рекомендации для отдела маркетинга
# 
# Ход исследования: <a id="soder"></a>
# 
# [1. Выгрузить данные и провести предобработку данных](#shag1)
# 
# 1.1. Импорт библиотек
# 
# 1.2. загрузка таблиц
# 
# 1.3. Функция для просмотра содержимого таблиц
# 
# 1.4. Таблица с информацией о посещениях сайта
# 
# 1.4.1. Корректировка названий столбцов
# 
# 1.4.2. Изменение типов данных
# 
# 1.4.3. Проверка на неявные дубликаты
# 
# 1.5. Таблица с информацией о зпказах
# 
# 1.5.1. Корректировка названий столбцов
# 
# 1.5.2. Изменение типов данных
# 
# 1.6. Таблица с информацией о расхлдах на рекламу
# 
# 1.6.1. Корректировка названий столбцов
# 
# 1.6.2. Изменение типовданных
# 
# 1.6.3. Проверка на неявные дубликаты
# 
# 1.7. Вывод по разделу
# 
# [2. Задать необходимые для исследования функции](#shag3)
# 
# 2.1. get_profiles() — для создания профилей пользователей
# 
# 2.2. get_retention() — для подсчёта удержания
# 
# 2.3. get_conversion() — для подсчёта конверсии
# 
# 2.4. get_ltv() — для подсчёта пожизненной ценности клиента
# 
# 2.5. filter_data() — для сглаживания данных
# 
# 2.6. plot_retention() — для построения графика удержания
# 
# 2.7. plot_conversion() — для построения графика конверсии
# 
# 2.8  plot_ltv_roi() — для визуализации пожизненной ценности клиента и коэффициента рентабельности инвестиций
# 
# 2.9. Сделать общий вывод по разделу
# 
# 
# [3. Провести исследовательский анализ данных](#shag3)
# 
# 3.1. Составить профили пользователей и определим их минимальную и максимальную даты привлечения
# 
# 3.2. Построить таблицу, отражающую количество пользователей и долю платящих из каждой страны
# 
# 3.3. Построить таблицу, отражающую количество пользователей и долю платящих для каждого устройства
# 
# 3.4. Построить таблицу, отражающую количество пользователей и долю платящих для каждого канала привлечения
# 
# 3.5. Сделать общий вывод по разделу
# 
# 
# [4. Маркетинг](#shag4)
# 
# 4.1. Посчитать общую сумму расходов на маркетинг
# 
# 4.2. Выяснить как траты распределены по рекламным источникам, то есть сколько денег потратили на каждый источник
# 
# 4.3. Построить график с визуализацией динамики изменения расходов во времени по неделям по каждому источнику
# 
# 4.4. Визуализировать динамику изменения расходов во времени по месяцам по каждому источнику
# 
# 4.5. Узнать, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника
# 
# 4.6. Проверить, почему наблюдаются резкие скачки стоимости привлечения пользователей в TipTop
# 
# 4.7. Сделать общий вывод по разделу
# 
# [5. Оцените окупаемость рекламы](#shag5)
# 
# 5.1. Проанализировать окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI
# 
# 5.2. Проверить конверсию и удержание пользователей, а так же динамику их изменения
# 
# 5.3. Проанализировать окупаемость рекламы с разбивкой по устройствам
# 
# 5.4. Проанализировать окупаемость рекламы с разбивкой по странам
# 
# 5.5. Проанализировать окупаемость рекламы с разбивкой по рекламным каналам
# 
# 5.6. Проанализировать конверсию и её динамику с разбивкой по устройствам
# 
# 5.7. Проанализировать конверсию и её динамику с разбивкой по странам
# 
# 5.8. Проанализировать конверсию и её динамику с разбивкой по рекламным каналам
# 
# 5.9. Проанализировать удержание и его динамику с разбивкой по устройствам
# 
# 5.10. Проанализировать удержание и его динамику с разбивкой по странам
# 
# 5.11. Проанализировать удержание и его динамику с разбивкой по рекламным каналам
# 
# 5.12. Сравнить окупаемость рекламы на разных каналах в разных странах
# 
# 5.13. Окупается ли реклама, направленная на привлечение пользователей в целом?
# 
# 5.14. Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
# 
# 5.15. Чем могут быть вызваны проблемы окупаемости?
# 
# 5.16. Сделать общий вывод по разделу
# 
# [6. Сделать общий вывод](#shag6)
# 

# ### Загрузите данные и подготовьте их к анализу <a id="shag1"></a>  
# [К содержанию](#soder)

# #### Импортируем необходимые библиотеки.

# In[85]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np


# #### Прочитаем предложенные таблицы

# In[86]:


visits = pd.read_csv('/datasets/visits_info_short.csv')


# In[87]:


orders = pd.read_csv('/datasets/orders_info_short.csv')


# In[88]:


costs = pd.read_csv('/datasets/costs_info_short.csv')


# #### Зададим функцию для предварительного просмотра содержимого таблиц. 

# In[89]:


def load(df):
        print ('Первые 20 строк таблицы:')
        print ()
        display (df.head(20))
        print ()
        print ('Иноформация о таблице:')
        print ()
        display (df.info())
        print ()
        print ('Количество дубликатов в таблице:')
        print ()
        display (df.duplicated().sum())
        print ()
        print ('Количество пропусков в таблице:')
        print ()
        display (df.isna().sum())
        print ()
        print ('Процентное соотношение пропусков к общему числу значений для каждого столбца')
        print ()
        display (pd.DataFrame(round(df.isna().mean()*100,)).style.background_gradient('coolwarm'))


# #### Рассмотрим таблицу с информацией о посещениях сайта.

# In[90]:


load(visits)


# В таблице содержится информация о посещениях сайта. Таблица состоит из 6 столбцов и 309901 строк.
# 
# Структура visits:
# 
# User Id — уникальный идентификатор пользователя (целочисленный тип данных),
# 
# Region — страна пользователя (объектный тип данных),
# 
# Device — тип устройства пользователя (объектный тип данных),
# 
# Channel — идентификатор источника перехода (объектный тип данных),
# 
# Session Start — дата и время начала сессии (объектный тип данных),
# 
# Session End — дата и время окончания сессии (объектный тип данных).
# 
# Названия столбцов следует привести к нижнему регистру.
# 
# Столбцы с датой и временем начала и окончания сессии необходимо привести к типу datetime.
# 
# Пропусков и явных дубликатов не обнаружено.

# ##### Приведем названия столбцов таблицы visits к нижнему регистру и изменим названия некотрых столбцов.

# In[91]:


visits.columns = visits.columns.str.lower().str.replace(' ', '_')


# ##### Приведём к типу datetime столбцы с датой и временем начала и окончания сессии.

# In[92]:


visits['session_start'] = pd.to_datetime(visits['session_start'])


# In[93]:


visits['session_end'] = pd.to_datetime(visits['session_end'])


# ##### Проверим столбцы Region, Device и Channel на неявные дубликаты.

# In[94]:


visits['region'].value_counts()


# In[95]:


visits['device'].value_counts()


# In[96]:


visits['channel'].value_counts()


# Неявных дубликатов не выявлено.

# #### Рассмотрим таблицу с информацией о заказах.

# In[97]:


load(orders)


# В таблице содержится информация о заказах. Таблица состоит из 3 столбцов и 340212 строк.
# 
# Структура orders:
# 
# User Id — уникальный идентификатор пользователя (целочисленный тип данных),
# 
# Event Dt — дата и время покупки (объектный тип данных),
# 
# Revenue — сумма заказа (вещественный тип данных).
# 
# Названия столбцов следует привести к нижнему регистру.
# 
# Столбец с датой и временем покупки необходимо привести к типу datetime.
# 
# Пропусков и явных дубликатов не обнаружено.

# ##### Приведем названия столбцов таблицы orders к нижнему регистру и изменим названия некотрых столбцов.

# In[98]:


orders.columns = orders.columns.str.lower().str.replace(' ', '_')


# ##### Приведём к типу datetime столбцы  с датой и временем покупки.

# In[99]:


orders['event_dt'] = pd.to_datetime(orders['event_dt'])


# #### Рассмотрим таблицу с информацией о расходах на рекламу.

# In[100]:


load(costs)


# В таблице содержится информация о расходах на рекламу. Таблица состоит из 3 столбцов и 1800 строк.
# 
# Структура costs:
# dt — дата проведения рекламной кампании (объектный тип данных),
# 
# Channel — идентификатор рекламного источника (объектный тип данных),
# 
# costs — расходы на эту кампанию (вещественный тип данных).
# 
# Названия столбцов следует привести к нижнему регистру.
# 
# Столбец с датой проведения рекламной кампании необходимо привести к типу datetime.
# 
# Пропусков и явных дубликатов не обнаружено.

# ##### Приведем названия столбцов таблицы costs к нижнему регистру.

# In[101]:


costs.columns = costs.columns.str.lower().str.replace(' ', '_')


# ##### Приведём к типу datetime столбец с датой проведения рекламной кампании.

# In[102]:


costs['dt'] = pd.to_datetime(costs['dt'])


# ##### Проверим столбец channel на неявные дубликаты.

# In[103]:


visits['channel'].value_counts()


# Неявных дубликатов не выявлено.

# #### Общий вывод

# В таблице visits содержится информация о посещениях сайта. Таблица состоит из 6 столбцов и 309901 строк.
# 
# Структура visits:
# 
# User Id — уникальный идентификатор пользователя (целочисленный тип данных),
# 
# Region — страна пользователя (объектный тип данных),
# 
# Device — тип устройства пользователя (объектный тип данных),
# 
# Channel — идентификатор источника перехода (объектный тип данных),
# 
# Session Start — дата и время начала сессии (объектный тип данных),
# 
# Session End — дата и время окончания сессии (объектный тип данных).
# 
# Названия столбцов приведены к нижнему регистру.
# 
# Столбцы с датой и временем начала и окончания сессии приведены к типу datetime.
# 
# Пропусков, явных и неявных дубликатов не обнаружено.
# 
# В таблице orders содержится информация о заказах. Таблица состоит из 3 столбцов и 340212 строк.
# 
# Структура orders:
# 
# User Id — уникальный идентификатор пользователя (целочисленный тип данных),
# 
# Event Dt — дата и время покупки (объектный тип данных),
# 
# Revenue — сумма заказа (вещественный тип данных).
# 
# Названия столбцов приведены к нижнему регистру.
# 
# Столбец с датой и временем покупки приведены к типу datetime.
# 
# Пропусков и явных дубликатов не обнаружено.
# 
# В таблице costs содержится информация о расходах на рекламу. Таблица состоит из 3 столбцов и 1800 строк.
# 
# Структура costs: dt — дата проведения рекламной кампании (объектный тип данных),
# 
# Channel — идентификатор рекламного источника (объектный тип данных),
# 
# costs — расходы на эту кампанию (вещественный тип данных).
# 
# Названия столбцов приведены к нижнему регистру.
# 
# Столбец с датой проведения рекламной кампании приведены к типу datetime.
# 
# Пропусков, явных и неявных дубликатов не обнаружено.

# ### Задайте функции для расчёта и анализа LTV, ROI, удержания и конверсии. <a id="shag2"></a>  
# [К содержанию](#soder)

# #### Зададим функцию для создания профилей пользователей get_profiles().

# In[104]:


def get_profiles(visits, 
                 orders, 
                 costs
                ):
    profiles = (
        visits.sort_values(by=['user_id', 'session_start'])
        .groupby('user_id')
        .agg(
            {
                'session_start': 'first',
                'channel': 'first',
                'device': 'first',
                'region': 'first'
            }
        )
        .rename(columns={'session_start': 'first_ts'})
        .reset_index()
    )

    profiles['first_ts'] = pd.to_datetime(profiles['first_ts'])
    profiles['dt'] = profiles['first_ts'].dt.date  
    profiles['month'] = profiles['first_ts'].dt.month
    profiles['week'] = profiles['first_ts'].dt.isocalendar().week

    profiles['payer'] = profiles['user_id'].isin(orders['user_id'].unique())

    new_users = (
        profiles.groupby(['dt', 'channel'])
        .agg({'user_id': 'nunique'})
        .rename(columns={'user_id': 'unique_users'})
        .reset_index()
    )

    costs['dt'] = costs['dt'].dt.date
    
    costs = costs.merge(new_users, on=['dt', 'channel'], how='left')

    costs['acquisition_cost'] = costs['costs'] / costs['unique_users']

    profiles = profiles.merge(
        costs[['dt', 'channel', 'acquisition_cost']],
        on=['dt', 'channel'],
        how='left',
    )

    profiles['acquisition_cost'] = profiles['acquisition_cost'].fillna(0)

    return profiles


# #### Зададим функцию для подсчёта Retention Rate get_retention().

# In[105]:


def get_retention(profiles, 
                  sessions, 
                  observation_date, 
                  horizon_days, 
                  dimensions = [], 
                  ignore_horizon = False
                 ):
    
    dimensions = ['payer'] + dimensions
    
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(days = horizon_days - 1)
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    result_raw = result_raw.merge(sessions[['user_id', 'session_start']], on = 'user_id', how = 'left')
    result_raw['lifetime'] = (result_raw['session_start'] - result_raw['first_ts']).dt.days
    
    def group_by_dimensions(df, dims, horizon_days):     
        result = df.pivot_table(index = dims, columns = 'lifetime', values = 'user_id', aggfunc = 'nunique')     
        cohort_sizes = df.groupby(dims).agg({'user_id': 'nunique'}).rename(columns = {'user_id': 'cohort_size'}) 
        result = cohort_sizes.merge(result, on = dims, how = 'left').fillna(0)                                   
        result = result.div(result['cohort_size'], axis = 0)                                                     
        result = result[['cohort_size'] + list(range(horizon_days))]                                             
        result['cohort_size'] = cohort_sizes                                                                     
        return result
    
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)
    
    result_in_time = group_by_dimensions(result_raw, dimensions + ['dt'], horizon_days)
    
    return result_raw, result_grouped, result_in_time


# #### Зададим функцию для подсчёта конверсии get_conversion().

# In[106]:


def get_conversion(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False
):

    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    first_purchases = (
        purchases.sort_values(by=['user_id', 'event_dt'])
        .groupby('user_id')
        .agg({'event_dt': 'first'})
        .reset_index()
    )
    
    result_raw = result_raw.merge(
        first_purchases[['user_id', 'event_dt']], on='user_id', how='left'
    )

    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days

    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users' 
        dimensions = dimensions + ['cohort']

    def group_by_dimensions(df, dims, horizon_days):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        result = result.fillna(0).cumsum(axis = 1)
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    if 'cohort' in dimensions: 
        dimensions = []

    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    return result_raw, result_grouped, result_in_time


# #### Зададим функцию для подсчёта LTV get_ltv().

# In[107]:


def get_ltv(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False
):
 
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')
    
    result_raw = result_raw.merge(
        purchases[['user_id', 'event_dt', 'revenue']], on='user_id', how='left'
    )
    
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days
    
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']

    def group_by_dimensions(df, 
                            dims, 
                            horizon_days
                           ):
        
        result = df.pivot_table(
            index=dims, columns='lifetime', values='revenue', aggfunc='sum'
        )
        
        result = result.fillna(0).cumsum(axis=1)
        
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        
        result = result.div(result['cohort_size'], axis=0)
        
        result = result[['cohort_size'] + list(range(horizon_days))]
        
        result['cohort_size'] = cohort_sizes

        cac = df[['user_id', 'acquisition_cost'] + dims].drop_duplicates()

        cac = (
            cac.groupby(dims)
            .agg({'acquisition_cost': 'mean'})
            .rename(columns={'acquisition_cost': 'cac'})
        )

        
        roi = result.div(cac['cac'], axis=0)

        roi = roi[~roi['cohort_size'].isin([np.inf])]

        roi['cohort_size'] = cohort_sizes

        roi['cac'] = cac['cac']

        roi = roi[['cohort_size', 'cac'] + list(range(horizon_days))]

        return result, roi

    result_grouped, roi_grouped = group_by_dimensions(result_raw, 
                                                      dimensions, 
                                                      horizon_days
                                                     )

   
    if 'cohort' in dimensions:
        dimensions = []

    result_in_time, roi_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    return (
        result_raw,  
        result_grouped,  
        result_in_time,  
        roi_grouped,  
        roi_in_time
    )


# #### Зададим функцию для сглаживания данных filter_data().

# In[108]:


def filter_data(df, 
                window
               ):
    for column in df.columns.values:
        df[column] = df[column].rolling(window).mean() 
    return df


# #### Зададим функцию для построения графика Retention Rate plot_retention().

# In[109]:


def plot_retention(retention, 
                   retention_history, 
                   horizon, 
                   window=7
                  ):

    plt.figure(figsize=(15, 10))

    retention = retention.drop(columns=['cohort_size', 0])
    
    retention_history = retention_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]
 
    if retention.index.nlevels == 1:
        retention['cohort'] = 'All users'
        retention = retention.reset_index().set_index(['cohort', 'payer'])
    
    ax1 = plt.subplot(2, 2, 1)
    retention.query('payer == True').droplevel('payer').T.plot(
        grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание платящих пользователей')

    
    ax2 = plt.subplot(2, 2, 2, sharey=ax1)
    retention.query('payer == False').droplevel('payer').T.plot(
        grid=True, ax=ax2)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание неплатящих пользователей')

    
    ax3 = plt.subplot(2, 2, 3)
    columns = [
        name
        for name in retention_history.index.names
        if name not in ['dt', 'payer']]
    
    filtered_data = retention_history.query('payer == True').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания платящих пользователей на {}-й день'.format(
            horizon))

    ax4 = plt.subplot(2, 2, 4, sharey=ax3)
    filtered_data = retention_history.query('payer == False').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
    filter_data(filtered_data, window).plot(grid=True, ax=ax4)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания неплатящих пользователей на {}-й день'.format(
            horizon))
    
    plt.tight_layout()
    plt.show()


# #### Зададим функцию для построения графика конверсии plot_conversion().

# In[110]:


def plot_conversion(conversion, conversion_history, horizon, window=7):
    
    plt.figure(figsize=(15, 5))

    conversion = conversion.drop(columns=['cohort_size'])
    
    conversion_history = conversion_history.drop(columns=['cohort_size'])[
        [horizon - 1]]

    ax1 = plt.subplot(1, 2, 1)
    conversion.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Конверсия пользователей')

    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    columns = [
        name for name in conversion_history.index.names if name not in ['dt']]
    filtered_data = conversion_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика конверсии пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show()


# #### Зададим функцию для визуализации LTV и ROI plot_ltv_roi.

# In[111]:


def plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon, window=7):

    plt.figure(figsize=(20, 10))

    ltv = ltv.drop(columns=['cohort_size'])
    
    ltv_history = ltv_history.drop(columns=['cohort_size'])[[horizon - 1]]

    cac_history = roi_history[['cac']]

    roi = roi.drop(columns=['cohort_size', 'cac'])
    
    roi_history = roi_history.drop(columns=['cohort_size', 'cac'])[
        [horizon - 1]]

    ax1 = plt.subplot(2, 3, 1)
    ltv.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('LTV')

    ax2 = plt.subplot(2, 3, 2, sharey=ax1)
    columns = [name for name in ltv_history.index.names if name not in ['dt']]
    filtered_data = ltv_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика LTV пользователей на {}-й день'.format(horizon))

    ax3 = plt.subplot(2, 3, 3, sharey=ax1)
    columns = [name for name in cac_history.index.names if name not in ['dt']]
    filtered_data = cac_history.pivot_table(
        index='dt', columns=columns, values='cac', aggfunc='mean')
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика стоимости привлечения пользователей')

    ax4 = plt.subplot(2, 3, 4)
    roi.T.plot(grid=True, ax=ax4)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('ROI')

    ax5 = plt.subplot(2, 3, 5, sharey=ax4)
    columns = [name for name in roi_history.index.names if name not in ['dt']]
    filtered_data = roi_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean')
    filter_data(filtered_data, window).plot(grid=True, ax=ax5)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.xlabel('Дата привлечения')
    plt.title('Динамика ROI пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show()


# #### Общий вывод

# Заданы функции для вычисления значений метрик:
# 
# - get_profiles() — для создания профилей пользователей
# - get_retention() — для подсчёта Retention Rate
# - get_conversion() — для подсчёта конверсии
# - get_ltv() — для подсчёта LTV
# 
# А также функции для построения графиков:
# 
# - filter_data() — для сглаживания данных,
# - plot_retention() — для построения графика Retention Rate,
# - plot_conversion() — для построения графика конверсии,
# - plot_ltv_roi — для визуализации LTV и ROI.

# ### Исследовательский анализ данных <a id="shag3"></a>  
# [К содержанию](#soder)

# #### Составим профили пользователей и определим  их минимальную и максимальную даты привлечения.

# In[112]:


profiles = get_profiles(visits, orders, costs)
profiles


# In[113]:


min_attraction_date = profiles['dt'].min()
max_attraction_date = profiles['dt'].max()
print('Минимальная дата привлечения пользователей:', min_attraction_date)
print('Максимальная дата привлечения пользователей:', max_attraction_date)


# Вывод:
# 
# В таблице profiles содержится информация о 150 008 профилей пользователей, с датами первого посещения сайта, страной, каналом и устройством входа, в последных столбцах выведен месяц первого вхада, информация о том, покупал ли пользователь что-либо в течение исследуемого периода и расходы на привлечение пользователя. 
# 
# Минимальная дата привлечения пользователей: 2019-05-01. 
# 
# Максимальная дата привлечения пользователей: 2019-10-27.
# 
# Они соответствуют границе исследуемого периода.

# #### Построим таблицу, отражающую количество пользователей и долю платящих из каждой страны.

# In[114]:


payers_by_region = (
    profiles.groupby(
    'region').agg(
    {'user_id' : 'nunique', 
     'payer' : 'sum'}
).merge((
    (profiles.groupby(
        'region').agg(
        {'payer' : 'mean'})*100)
    ),
    on = 'region', 
    how='left')
).rename(
    columns={
        'user_id' : 'users_count', 
        'payer_x' : 'payers_count', 
        'payer_y' : 'payers_pers'}
).sort_values(
    by='payers_pers', 
    ascending=False)

payers_by_region.round(2)


# In[115]:


plt.figure(figsize=(15, 7))
ax1 = plt.subplot(1, 2, 1)
payers_by_region.plot(kind='pie',
                      y = 'users_count',
                      subplots=True,  
                      autopct ='%1.2f%%', 
                      legend=False,
                      ylabel='',
                      ax=ax1)
plt.title('Пользователи по странам')

ax2 = plt.subplot(1, 2, 2)
payers_by_region.plot(kind='pie',
                     y = 'payers_count',
                     subplots=True,  
                     autopct ='%1.2f%%', 
                     legend=False, 
                     ylabel='',
                     ax=ax2)
plt.title('Платящие пользователи по странам')

plt.show()


# Вывод:
# 
# Больше всего зарегистрированных пользователей приходится на США, далее Франция, Англия и Германия.
# 
# Процент платящих пользователей больше всего также в США.
# 
# Отношение платящих пользователей к их общему количеству больше всего в США, далее Германия, Англия и Франция.
# 
# Пользователей из Германии меньше всего, но при этом их "качество" выше чем у пользователей из Франции и Англии.

# #### Построим таблицу, отражающую количество пользователей и долю платящих  для каждого устройства.

# In[116]:


payers_by_device = (
    profiles.groupby(
    'device').agg(
    {'user_id' : 'nunique', 
     'payer' : 'sum'}
).merge((
    (profiles.groupby(
        'device').agg(
        {'payer' : 'mean'})*100)
    ),
    on = 'device', 
    how='left')
).rename(
    columns={
        'user_id' : 'users_count', 
        'payer_x' : 'payers_count', 
        'payer_y' : 'payers_pers'}
).sort_values(
    by='payers_pers', 
    ascending=False)

payers_by_device.round(2)


# In[117]:


plt.figure(figsize=(15, 7))
ax1 = plt.subplot(1, 2, 1)
payers_by_device.plot(kind='pie',
                      y = 'users_count',
                      subplots=True,  
                      autopct ='%1.2f%%', 
                      legend=False,
                      ylabel='',
                      ax=ax1)
plt.title('Пользователи по устройству')

ax2 = plt.subplot(1, 2, 2)
payers_by_device.plot(kind='pie',
                     y = 'payers_count',
                     subplots=True,  
                     autopct ='%1.2f%%', 
                     legend=False, 
                     ylabel='',
                     ax=ax2)
plt.title('Платящие пользователи по устройству')

plt.show()


# Вывод:
# 
# Больше всего зарегистрированных пользователей заходят с Айфонов, далее Андроид, ПК и Мак.
# 
# Процент платящих пользователей больше всего также на Айфонах.
# 
# Отношение платящих пользователей к их общему количеству больше всего на Мак, далее Айфон, Андроид и ПК.
# 
# Пользователей Мак меньше всего, но при этом их "качество" выше чем у всех остальных пользователей.

# #### Построим таблицу, отражающую количество пользователей и долю платящих  для каждого канала привлечения.

# In[118]:


payers_by_channel = (
    profiles.groupby(
    'channel').agg(
    {'user_id' : 'nunique', 
     'payer' : 'sum'}
).merge((
    (profiles.groupby(
        'channel').agg(
        {'payer' : 'mean'})*100)
    ),
    on = 'channel', 
    how='left')
).rename(
    columns={
        'user_id' : 'users_count', 
        'payer_x' : 'payers_count', 
        'payer_y' : 'payers_pers'}
).sort_values(
    by='payers_pers', 
    ascending=False)

payers_by_channel.round(2)


# In[119]:


plt.figure(figsize=(17, 9))
ax1 = plt.subplot(1, 2, 1)
payers_by_channel.sort_values(by='channel').plot(kind='bar',
                                                y = 'users_count',
                                                subplots=True,  
                                                fontsize=10,
                                                legend=False,
                                                ylabel='',
                                                ax=ax1)
plt.title('Пользователи по каналу привлечения')

ax2 = plt.subplot(1, 2, 2)
payers_by_channel.sort_values(by='channel').plot(kind='bar',
                                                 y = 'payers_count',
                                                 subplots=True,
                                                 legend=False,
                                                 fontsize=10,
                                                 ylabel='',
                                                 ax=ax2)
plt.title('Платящие пользователи по каналу привлечения')

plt.show()


# Вывод:
# 
# Больше всего зарегистрированных пользователей (кроме тех кто приходят сами) приходят из FaceBoom, далее идет TipTop и другие.
# 
# Процент платящих пользователей больше всего от каналов FaceBoom и TipTop.
# 
# Отношение платящих пользователей к их общему количеству больше всего на FaceBoom, далее AdNonSense, lambdaMediaAds и TipTop.
# 
# Пользователей AdNonSense и lambdaMediaAds меньше всего, но при этом их "качество" выше чем у всех остальных пользователей (кроме FaceBoom).

# #### Общий вывод

# В таблице profiles содержится информация о 150 008 профилей пользователей, с датами первого посещения сайта, страной, каналом и устройством входа, в последных столбцах выведен месяц первого вхада и информация о том, покупал ли пользователь что-либо в течение исследуемого периода. 
# 
# Минимальная дата привлечения пользователей: 2019-05-01. 
# 
# Максимальная дата привлечения пользователей: 2019-10-27.
# 
# Они соответствуют границе исследуемого периода.
# 
# Больше всего зарегистрированных пользователей приходится на США, далее Франция, Англия и Германия.
# 
# Процент платящих пользователей больше всего также в США.
# 
# Отношение платящих пользователей к их общему количеству больше всего в США, далее Германия, Англия и Франция.
# 
# Пользователей из Германии меньше всего, но при этом их "качество" выше чем у пользователей из Франции и Англии.
# 
# Больше всего зарегистрированных пользователей заходят с Айфонов, далее Андроид, ПК и Мак.
# 
# Процент платящих пользователей больше всего также на Айфонах.
# 
# Отношение платящих пользователей к их общему количеству больше всего на Мак, далее Айфон, Андроид и ПК.
# 
# Пользователей Мак меньше всего, но при этом их "качество" выше чем у всех остальных пользователей.
# 
# Больше всего зарегистрированных пользователей (кроме тех кто приходят сами) приходят из FaceBoom их около 19%, далее идет TipTop (13%) и другие.
# 
# Процент платящих пользователей больше всего от каналов FaceBoom и TipTop.
# 
# Отношение платящих пользователей к их общему количеству больше всего на FaceBoom, далее AdNonSense, lambdaMediaAds и TipTop.
# 
# Пользователей AdNonSense и lambdaMediaAds меньше всего, но при этом их "качество" выше чем у всех остальных пользователей (кроме FaceBoom).

# ### Маркетинг <a id="shag4"></a>  
# [К содержанию](#soder)

# #### Посчитаем общую сумму расходов на маркетинг.

# In[120]:


costs['costs'].sum()


# Вывод
# 
# Всего за исследуемый период на рекламу было потрачено 105497.3 $.

# #### Выясним как траты распределены по рекламным источникам, то есть сколько денег потратили на каждый источник.

# In[121]:


costs.groupby('channel').agg({'costs' : 'sum'}).sort_values(by = 'costs', ascending=False)


# In[122]:


costs.groupby('channel').agg({'costs' : 'sum'}).sort_values(by = 'costs', ascending=False).plot(kind='bar',
                     y = 'costs',
                     subplots=True, 
                     legend=False,
                     figsize=(17, 7),                                                                           
                     ylabel='')
plt.title('Распределение трат по рекламным источникам')
plt.show()


# Вывод:
# 
# Больше всего потратили на рекламу в TipTop и FaceBoom.
# 
# Меньше всего на YRabbit, MediaTornado и  lambdaMediaAds.

# #### Построим график с визуализацией динамики изменения расходов во времени по неделям по каждому источнику.

# In[123]:


costs_monthweek = costs.copy()


# In[124]:


costs_monthweek['dt'] = pd.to_datetime(costs_monthweek['dt'])
costs_monthweek['week'] = costs_monthweek['dt'].dt.isocalendar().week
costs_monthweek['month'] = costs_monthweek['dt'].dt.month

weekly_spending = costs_monthweek.pivot_table(index = 'week', columns = 'channel', values = 'costs', aggfunc = 'sum')


# In[125]:


weekly_spending


# In[126]:


weekly_spending.plot(grid=True, figsize=(17, 10))
plt.xlabel('Неделя')
plt.ylabel('Затраты на рекламу')
plt.title('Динамика изменения расходов на рекламу по неделям по каждому источнику')
plt.show()


# In[127]:


plt.figure(figsize=(17, 10))
sns.heatmap(
    weekly_spending.T,
    annot=True,
    fmt='.0f'
)
plt.xlabel('Неделя')
plt.ylabel('Канал')
plt.title('Динамика изменения расходов на рекламу по неделям по каждому источнику')
plt.show()


# Вывод
# 
# Расходы на рекламу в TipTop и FaceBoom постоянно растут, пики о обоих пришлись на 39 неделю года.
# 
# Расходы на остальные каналы привлечения пользователей примерно одинаковы, по ним не надлюдается значительных колебаний.

# #### Визуализируем динамику изменения расходов во времени по месяцам по каждому источнику.

# In[128]:


monthly_spending = costs_monthweek.pivot_table(index = 'month', columns = 'channel', values = 'costs', aggfunc = 'sum')


# In[129]:


monthly_spending


# In[130]:


monthly_spending.plot(grid=True, figsize=(17, 10))
plt.xlabel('Месяц')
plt.ylabel('Затраты на рекламу')
plt.title('Динамика изменения расходов на рекламу по месяцам по каждому источнику')
plt.show()


# In[131]:


plt.figure(figsize=(17, 10))
sns.heatmap(
    monthly_spending.T,
    annot=True, 
    fmt='.1f'
)
plt.xlabel('Месяц')
plt.ylabel('Канал')
plt.title('Динамика изменения расходов на рекламу по месяцам по каждому источнику')
plt.show()


# Вывод
# 
# Расходы на рекламу в TipTop и FaceBoom постоянно растут, пик стоимости рекламы в TipTop пришелся на сентябрь.
# 
# Расходы на остальные каналы привлечения пользователей примерно одинаковы, по ним не надлюдается значительных колебаний.

# #### Узнаем, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника.

# In[48]:


profiles.pivot_table(
    index='channel', 
    values='acquisition_cost', 
    aggfunc='mean').sort_values(
    by = 'acquisition_cost', 
    ascending=False).round(2)


# In[49]:


profiles.pivot_table(
    index='dt', columns='channel', values='acquisition_cost', aggfunc='mean'
).plot(grid=True, figsize=(17, 10))
plt.ylabel('CAC, $')
plt.xlabel('Дата привлечения')
plt.title('Динамика САС по каналам привлечения')
plt.show()


# Вывод
# 
# В среднем самым дорогим источником привлечения пользователей является TipTop, средняя стоимость привлечения пользователей окло 2,8\\$. 
# 
# Стоимость привлечения постоянно растет, начиная с 1\\$ в мае до 3,5\\$ в сентябре-октябре. График динамики САС внешне напоминает лестницу, примерно в середине мая, двадцатых числах июня, конце июля и начале сентября наблюдаются резкие скачки стоимости привлечения пользователей, этому может быть два объяснения: в эти даты проводили рекламные кампании, или по каким-то причинам снижалось количество привлеченных клиентов.
# 
# Проверим это далее.
# 
# Привлечение клиентов через FaceBoom и AdNonSense обходится  примерно в 1\\$, эти графики выглядят более стабильными. 

# #### Проверим, почему наблюдаются резкие скачки стоимости привлечения пользователей в TipTop.

# In[50]:


profiles.query('channel == "TipTop"').groupby('month').agg({'user_id' : 'count', 'acquisition_cost' : 'sum'})


# In[51]:


plt.figure(figsize=(17, 9))
ax1 = plt.subplot(1, 2, 1)
profiles.query(
    'channel == "TipTop"').groupby(
    'month').agg({'user_id' : 'count', 'acquisition_cost' : 'sum'}
                ).plot(y = 'user_id',
                      subplots=True,
                      fontsize=7,
                      legend=False,
                      ylabel='Количество привлеченных пользователей',
                      ax=ax1)
plt.title('')

ax2 = plt.subplot(1, 2, 2)
profiles.query(
    'channel == "TipTop"').groupby(
    'month').agg({'user_id' : 'count', 'acquisition_cost' : 'sum'}
                ).plot(y = 'acquisition_cost',
                     subplots=True,
                     legend=False,
                     fontsize=7,
                     ylabel='Затраты на рекламу',
                     ax=ax2)
plt.title('')

plt.show()


# In[52]:


profiles.query('channel == "TipTop"').groupby('week').agg({'user_id' : 'count', 'acquisition_cost' : 'sum'})


# In[53]:


plt.figure(figsize=(17, 9))
ax1 = plt.subplot(1, 2, 1)
profiles.query(
    'channel == "TipTop"').groupby(
    'week').agg({'user_id' : 'count', 'acquisition_cost' : 'sum'}
                ).plot(y = 'user_id',
                      subplots=True,
                      fontsize=7,
                      legend=False,
                      ylabel='Количество привлеченных пользователей',
                      ax=ax1)
plt.title('')

ax2 = plt.subplot(1, 2, 2)
profiles.query(
    'channel == "TipTop"').groupby(
    'week').agg({'user_id' : 'count', 'acquisition_cost' : 'sum'}
                ).plot(y = 'acquisition_cost',
                     subplots=True,
                     legend=False,
                     fontsize=7,
                     ylabel='Затраты на рекламу',
                     ax=ax2)
plt.title('')

plt.show()


# Вывод
# 
# Снижения количества привлеченных пользователей не наблюдается, скачки стоимости привлечения пользователей из TipTop обусловлены проведением ежемесячных рекламных кампаний.
# 
# Стоит проверить окупается ли повышение стоимости привлечения пользователей в следующем разделе.

# #### Общий вывод

# Всего за исследуемый период на рекламу было потрачено 105497,3\\$.
# 
# Больше всего потратили на рекламу в TipTop и FaceBoom.
# 
# Меньше всего на YRabbit, MediaTornado и lambdaMediaAds.
# 
# Расходы на рекламу в TipTop и FaceBoom постоянно растут, пики о обоих пришлись на 39 неделю года.
# 
# Расходы на остальные каналы привлечения пользователей примерно одинаковы, по ним не надлюдается значительных колебаний.
# 
# Расходы на рекламу в TipTop и FaceBoom постоянно растут, пик стоимости рекламы в TipTop пришелся на сентябрь.
# 
# Расходы на остальные каналы привлечения пользователей примерно одинаковы, по ним не наблюдается значительных колебаний.
# 
# В среднем самым дорогим источником привлечения пользователей является TipTop, средняя стоимость привлечения пользователей около 2,8\\$. 
# 
# Стоимость привлечения постоянно растет, начиная с 1\\$ в мае до 3,5\\$ в сентябре-октябре. График динамики САС внешне напоминает лестницу, примерно в середине мая, двадцатых числах июня, конце июля и начале сентября наблюдаются резкие скачки стоимости привлечения пользователей, этому может быть два объяснения: в эти даты проводили рекламные кампании, или по каким-то причинам снижалось количество привлеченных клиентов.
# 
# Проверим это далее.
# 
# Привлечение клиентов через FaceBoom и AdNonSense обходится  примерно в 1\\$, эти графики выглядят более стабильными. 
# 
# Снижения количества привлеченных пользователей не наблюдается, скачки стоимости привлечения пользователей из TipTop обусловлены проведением ежемесячных рекламных кампаний.
# 
# Стоит проверить окупается ли повышение стоимости привлечения пользователей в следующем разделе.

# ### Оцените окупаемость рекламы <a id="shag5"></a>  
# [К содержанию](#soder)

# #### Проанализируем окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI.

# In[54]:


observation_date = datetime(2019, 11, 1).date()
horizon_days = 14
profiles_ad = profiles.query('channel != "organic"')


# In[55]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad, orders, observation_date, horizon_days, dimensions=[]
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days)


# Вывод:
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,9\\$, LTV колеблется от 0,7\\$ до 1,1\\$ на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# При этом стоимость привлечения пользователей (САС) постоянно увеличивается с 0,7\\$ в мае до 1,3\\$ к концу октября.
# 
# Коэффициент рентабельности инвестиций меньше единицы, значит реклама не окупается, так же на графике динамики ROI наблидается падение с 1,4 в мае до 0,6 к концу октября, это говорит о снижении эффективности рекламной стратегии.

# #### Проверим конверсию и удержание пользователей, а так же динамику их изменения.

# In[56]:


conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles_ad, orders, observation_date, horizon_days)
plot_conversion(conversion_grouped, conversion_history, horizon_days) 


# Вывод:
# 
# Конверсия к 14 лайфтайму составляет около 8%.
# 
# В середине мая, конце июня, конце июля и конце августа конверсия четырнадцатого дня снижается до 7%.
# 
# В середине июня конверсия 14-го дня достигла пика 9,5%.
# 
# в сентябре-октябре колебания конверсии стали менее заметны.

# In[57]:


retention_raw, retention_grouped, retention_history = get_retention(
    profiles_ad, visits, observation_date, horizon_days)
plot_retention(retention_grouped, retention_history, horizon_days) 


# Вывод:
# 
# Удержание платящих пользователей к 14-му дню составляет окло 10%, для неплатящих этот параметр составляет 10% на 1-2 день, а к 14-му приближается к нулю.
# 
# Удержание платящих пользователей на 14-й день в середине мая было ниже всего и составляла окло 6%, также падения до 7,5% наблюдались в конце июня и конце июля. 
# 
# Самое высокое значение удержания платяжих пользователей на 14-й день наблюдалось в конце мая - начале июня, так же значение удержания превысио 15% в середине августа.
# 
# Удержание неплатящих пользвателей стабильно держится на уровне примерно 1% на протяжении всего исслудемого периода.

# #### Проанализируем окупаемость рекламы с разбивкой по устройствам.

# In[58]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad, orders, observation_date, horizon_days, dimensions=['device']
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days, window=14)


# Вывод:
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,9\\$ для пользователей android, mac и iphone, а для пользователей РС немного ниже, на уровне 0,85\\$, LTV колеблется от 0,6\\$ до 1\\$ для пользователей РС, пики LTV для пользователей android, mac и iphone колеблется от 0,7\\$ до 1,1\\$ на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# Стоимость привлечения пользователей (САС) ниже всего для пользователей РС, она постоянно увеличивается с 0,7\\$ в мае до 0,9\\$ к концу октября. Далее идут пользователи android (с 0,7\\$ в мае до 1\\$ к середине сентября, а далее произошло небольшое снижение к концу октября). Пользователи mac и iphone "стоят" дороже всего (с 0,8\\$ в мае до 1,5\\$ к концу октября).
#  
# Коэффициент рентабельности инвестиций для пользователей android, mac и iphone меньше единицы, значит реклама не окупается, а для пользователей РС после 11 лайфтайма становится немного ниже единицы и начинает окупаться.
# 
# Так же на графике динамики ROI для пользователей android, mac и iphone наблидается падение с 1,2 в мае до 0,6 к концу октября. У пользователей РС до сентября значения ROI почти всегда превышало 1, но в сентябре и октябре снизилось до 0,7, это говорит о снижении эффективности рекламной стратегии.

# #### Проанализируем окупаемость рекламы с разбивкой по странам.

# In[59]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad, orders, observation_date, horizon_days, dimensions=['region']
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days, window=14)


# Вывод:
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,8\\$ для пользователей из Германии, Англии и Франции, а для пользователей из США немного выше, и доходит до 1\\$, LTV у Американцев выше других на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# Стоимость привлечения пользователей (САС) в июне снизилась для пользователей из Германии, Англии и Франции с 0,6\\$ до 0,4\\$, а для пользователей из США она постоянно повышалась до 1,8\\$.
# 
# Коэффициент рентабельности инвестиций для пользователей Германии, Англии и Франции превышает единицу начиная с 4-5 дня, а для Американцев доходит только до 0,7, значит реклама в США не окупается.
# 
# Так же на графике динамики ROI для пользователей из Германии, Англии и Франции ROI стабильно выше 1, а для пользователей из Англии в середине июля и августа доходит до 2,25 и 2,6 соответственно. У пользователей из США значения ROI с мая до конца октября упали с 1,2 до 0,5 это говорит о снижении эффективности рекламной стратегии для США.

# #### Проанализируем окупаемость рекламы с разбивкой по рекламным каналам.

# In[60]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad, orders, observation_date, horizon_days, dimensions=['channel']
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days, window=14)


# In[61]:


roi[13].sort_values()


# Вывод:
# 
# LTV выше всего для пользователей lambdaMediaAds и TipTop и к концу втрой недели доходит до 1,5\\$ и 1,8\\$ соответственно.
# 
# Самые лучшие показатели LTV в динамике также у lambdaMediaAds и TipTop, причем lambdaMediaAds с начала сентября стабильно растет, а в начале июня LTV lambdaMediaAds достигал 3,2\\$.
# 
# Стоимость привлечения пользователей стабильна для всех каналов кроме TipTop, для него стоимость привлечения возрасла с 1\\$ до 3,5\\$.
# 
# Показатель ROI для большинства каналов превышает 1, реклама не окупается только для AdNonSense, FaceBoom и TipTop, самая лучшая окупаемость у lambdaMediaAds, MediaTornado и YRabbit.
# 
# Лучшее значение ROI показал канал YRabbit в середине июля, достигнув значения 7. Высокие показатели ROI дл YRabbit обусловлены в большей степени низкой стоимостью рекламы, чем высоким LTV.

# #### Проанализируем конверсию и её динамику с разбивкой по устройствам.

# In[62]:


conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles_ad, orders, observation_date, horizon_days, dimensions=['device'])
plot_conversion(conversion_grouped, conversion_history, horizon_days, window=14) 


# Вывод: 
# 
# Лучшие показатели конверсии у Mac, до 8 %. Худшие у PC, до 6,5 %.
# 
# В динамике конверсии 14-го заметен резкий провал у пользователей РС в конце июля до 5%, резкий скачок у пользователей andriod в 10-х числах сентября до 10,5% и резкий скачок у пользователей Mac в 20-х числах сентября до 10,5%.

# #### Проанализируем конверсию и её динамику с разбивкой по странам.

# In[63]:


conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles_ad, orders, observation_date, horizon_days, dimensions=['region'])
plot_conversion(conversion_grouped, conversion_history, horizon_days, window=14) 


# Вывод: 
# 
# Лучшие показатели конверсии у США, до 9,5%. Для остальных стран показатель примерно на одном уровне, около 5%.
# 
# В динамике конверсии 14-го для заметно снижение конверсии у Англии, Франции и Германии с 6% в мае до 5% в октябре, зато в США конверсия возрастала с 8% до 11% до середины июня, и сохраняется на уровне окло 9% на протяжении всего периода исследования.

# #### Проанализируем конверсию и её динамику с разбивкой по рекламным каналам.

# In[64]:


conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles_ad, orders, observation_date, horizon_days, dimensions=['channel'])
plot_conversion(conversion_grouped, conversion_history, horizon_days, window=14) 


# Вывод:
# 
# Самая высокая конверсия у канала FaceBoom, к концу второй недели она достигает 12%, далее идут AdNonSense, lambdaMediaAds и TipTop, из конверсия 14-го дня примерно на уровне 10%.
# 
# Динамика конверсии пользователей на 14-й день наиболее нестабильна для lambdaMediaAds, в середине июня и августа она достигает 16%, а в начале сентября снижается до 5%.

# #### Проанализируем удержание и его динамику с разбивкой по устройствам.

# In[65]:


retention_raw, retention_grouped, retention_history = get_retention(profiles, 
                  visits, 
                  observation_date, 
                  horizon_days, 
                  dimensions = ['device'], 
                  ignore_horizon = False
                 )
plot_retention(retention_grouped, retention_history, horizon_days) 


# Удержание платящих пользователей примерно одинаково для всех устройств и к 14 дню достигает 10% для iphohe, 12-13% для mac и android и 15% для PC.
# 
# Удержание неплатящих пользователей примерно одинаково для всех устройств и стремится к нулю.
# 
# В середине мая удержание платящих пользователей РС достигало 37%, так же неплохо себя показали РС и mac в конце мая и середине июля, тогда показатели удержания у обоих устройств достигали 25%.

# #### Проанализируем удержание и его динамику с разбивкой по странам.

# In[66]:


retention_raw, retention_grouped, retention_history = get_retention(profiles, 
                  visits, 
                  observation_date, 
                  horizon_days, 
                  dimensions = ['region'], 
                  ignore_horizon = False
                 )
plot_retention(retention_grouped, retention_history, horizon_days) 


# Удержание платящих пользователей примерно одинаково для стран Европы и к 14 дню достигает 15-20%. Удержание для пользователей из США к 14 дню достигает лишь 10%
# 
# Удержание неплатящих пользователей примерно одинаково для всех стран и стремится к нулю.

# #### Проанализируем удержание и его динамику с разбивкой по рекламным каналам.

# In[67]:


retention_raw, retention_grouped, retention_history = get_retention(profiles, 
                  visits, 
                  observation_date, 
                  horizon_days, 
                  dimensions = ['channel'], 
                  ignore_horizon = False
                 )
plot_retention(retention_grouped, retention_history, horizon_days) 


# Удержание платящих пользователей примерно одинаково для большинства каналов и к 14 дню достигает 20-25%. У MediaTomado и YRabbiit удержание 14 для достигает 10-12%. Хуже всего показывают себя AdNonSense и FaceBoom, их удержание близко к нулю.
# 
# Удержание неплатящих пользователей примерно одинаково для всех стран и стремится к нулю.

# #### Сравним, окупаемость рекламы на разных каналах в разных странах.

# In[68]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad.query('region == "United States"'), orders, observation_date, horizon_days, dimensions=['channel']
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days, window=14)
roi[13].sort_values()


# В США работают 5 рекламных платформ, три из них RocketSuperAds, MediaTornado и YRabbit хорошо окупаются, но TipTop и FaceBoom не окупаются из-за высокой стоимости привлечения пользователей.

# In[69]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad.query('region == "France"'), orders, observation_date, horizon_days, dimensions=['channel']
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days, window=14)
roi[13].sort_values()


# Во Франции работают 5 рекламных платформ, две из них LeapBob и lambdaMediaAds хорошо окупаются, еще две OppleCreativeMedia и WahooNetBannerокупаются немного хуже, и AdNonSense не окупается из-за высокой стоимости привлечения пользователей.

# In[70]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad.query('region == "Germany"'), orders, observation_date, horizon_days, dimensions=['channel']
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days, window=14)
roi[13].sort_values()


# В Германии работают 5 рекламных платформ, две из них LeapBob и lambdaMediaAds хорошо окупаются, еще две OppleCreativeMedia и WahooNetBanner окупаются немного хуже, и AdNonSense не окупается из-за высокой стоимости привлечения пользователей.

# In[71]:


ltv_raw, ltv, ltv_history, roi, roi_history = get_ltv(
    profiles_ad.query('region == "UK"'), orders, observation_date, horizon_days, dimensions=['channel']
)
plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon_days, window=14)
roi[13].sort_values()


# В Англии работают 5 рекламных платформ, две из них LeapBob и lambdaMediaAds хорошо окупаются, еще две OppleCreativeMedia и WahooNetBanner окупаются немного хуже, и AdNonSense не окупается из-за высокой стоимости привлечения пользователей.

# Для стран Европы эффективные рекламные каналы одни и те же, это LeapBob и lambdaMediaAds. LeapBob привлекает пользователей с невысокой пожизненной ценность, но затраты на этот рекламный канал очень малы, поэтому он хорошо окупается. lambdaMediaAds стоит дороже, но и привлекает более "качественных" пользователей.  AdNonSense не окупился ни в одной из стран Европы, от него следует отказаться.
# 
# Для США хорошую окупаемость показали RocketSuperAds, MediaTornado и YRabbit. Пожизненная ценность пользователей из этих каналов ниже чем у пользователей из TipTop, но при этом сама реклама на этих каналах стоит гораздо меньше.

# #### Окупается ли реклама, направленная на привлечение пользователей в целом?

# В целом реклама перестала окупаться начиая с июня.
# 
# Показатель ROI к 14 дню лоходит до отметки 0,8, это значит, что к 14 дню возвращается только 80% стоимости рекламы.

# #### Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?

# Негативное воздействие на окупаемость рекламы могут оказывать:
# - Устройства:
#     - mac и iphone, их ROI ниже всего, при том что доли платящих пользователей для этих устройств выше других
# - Страны:
#     - США, показатель ROI для этой страны самый низкий. Реклама в США перестала окупаться в конце мая, и с тех пор показатель ROI не превышает единицу и стабильно снижается из-за огромных затрат на рекламу в TipTop и высоких затрат на рекламу в FaceBoom. Усугубляет ситуацию, что доля платящих пользователей из США составляет 78% от общего числа платящих пользователей. 
# - Рекламные каналы:
#     - Каналы FaceBoom и TipTop (США). Несмотря на то, что эти два канала привлекают 60% платящих пользователей, их окупаемость находится на уровне 54% для TipTop и 74% для FaceBoom. 
#     - Канал AdNonSense (ЕС)

# #### Чем могут быть вызваны проблемы окупаемости?

# Проблемы окупаемости рекламы могут быть вызваны:
# 
# - Высокой стоимостью привлечения пользователей mac и iphone
# - Низкой конверсией пользователей РС
# - Низким удержанием пользователей из США
# - низким удержанием пользователей пришедших с каналов AdNonSense и FaceBoom
# - Низкой пожизненной стоимостью пользователей из Германии, Англии и Франции
# - Очень высокой стоимостью привлечения пользователей из США по каналу TipTop
# - Высокой стоимостью привлечения пользователей из ЕС по каналу AdNonSense
# - Низкой пожизненной стоимостью пользователей из каналов: OppleCreativeMedia, LeapBob, MediaTornado, YRabbit, AdNonSense, FaceBoom

# #### Общий вывод

# Пожизненная ценность клиента к концу второй недели составляет около 0,9\\$, LTV колеблется от 0,7\\$ до 1,1\\$ на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# При этом стоимость привлечения пользователей (САС) постоянно увеличивается с 0,7\\$ в мае до 1,3\\$ к концу октября.
# 
# Коэффициент рентабельности инвестиций меньше единицы, значит реклама не окупается, так же на графике динамики ROI наблидается падение с 1,4 в мае до 0,6 к концу октября, это говорит о снижении эффективности рекламной стратегии.
# 
# Конверсия к 14 лайфтайму составляет около 8%.
# 
# В середине мая, конце июня, конце июля и конце августа конверсия четырнадцатого дня снижается до 7%.
# 
# В середине июня конверсия 14-го дня достигла пика 9,5%.
# 
# в сентябре-октябре колебания конверсии стали менее заметны.
# 
# Удержание платящих пользователей к 14-му дню составляет окло 10%, для неплатящих этот параметр составляет 10% на 1-2 день, а к 14-му приближается к нулю.
# 
# Удержание платящих пользователей на 14-й день в середине мая было ниже всего и составляла окло 6%, также падения до 7,5% наблюдались в конце июня и конце июля.
# 
# Самое высокое значение удержания платяжих пользователей на 14-й день наблюдалось в конце мая - начале июня, так же значение удержания превысио 15% в середине августа.
# 
# Удержание неплатящих пользвателей стабильно держится на уровне примерно 1% на протяжении всего исследуемого периода.
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,9\\$ для пользователей android, mac и iphone, а для пользователей РС немного ниже, на уровне 0,85\\$, LTV колеблется от 0,6\\$ до 1\\$ для пользователей РС, пики LTV для пользователей android, mac и iphone колеблется от 0,7\\$ до 1,1\\$ на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# Стоимость привлечения пользователей (САС) ниже всего для пользователей РС, она постоянно увеличивается с 0,7\\$ в мае до 0,9\\$ к концу октября. Далее идут пользователи android (с 0,7\\$ в мае до 1\\$ к середине сентября, а далее произошло небольшое снижение к концу октября). Пользователи mac и iphone "стоят" дороже всего (с 0,8\\$ в мае до 1,5\\$ к концу октября).
#  
# Коэффициент рентабельности инвестиций для пользователей android, mac и iphone меньше единицы, значит реклама не окупается, а для пользователей РС после 11 лайфтайма становится немного ниже единицы и начинает окупаться.
# 
# Так же на графике динамики ROI для пользователей android, mac и iphone наблидается падение с 1,2 в мае до 0,6 к концу октября. У пользователей РС до сентября значения ROI почти всегда превышало 1, но в сентябре и октябре снизилось до 0,7, это говорит о снижении эффективности рекламной стратегии.
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,8\\$ для пользователей из Германии, Англии и Франции, а для пользователей из США немного выше, и доходит до 1\\$, LTV у Американцев выше других на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# Стоимость привлечения пользователей (САС) в июне снизилась для пользователей из Германии, Англии и Франции с 0,6\\$ до 0,4\\$, а для пользователей из США она постоянно повышалась до 1,8\\$.
# 
# Коэффициент рентабельности инвестиций для пользователей Германии, Англии и Франции превышает единицу начиная с 4-5 дня, а для Американцев доходит только до 0,7, значит реклама в США не окупается.
# 
# Так же на графике динамики ROI для пользователей из Германии, Англии и Франции ROI стабильно выше 1, а для пользователей из Англии в середине июля и августа доходит до 2,25 и 2,6 соответственно. У пользователей из США значения ROI с мая до конца октября упали с 1,2 до 0,5 это говорит о снижении эффективности рекламной стратегии для США.
# 
# LTV выше всего для пользователей lambdaMediaAds и TipTop и к концу втрой недели доходит до 1,5\\$ и 1,8\\$ соответственно.
# 
# Самые лучшие показатели LTV в динамике также у lambdaMediaAds и TipTop, причем lambdaMediaAds с начала сентября стабильно растет, а в начале июня LTV lambdaMediaAds достигал 3,2\\$.
# 
# Стоимость привлечения пользователей стабильна для всех каналов кроме TipTop, для него стоимость привлечения возрасла с 1\\$ до 3,5\\$.
# 
# Показатель ROI для большинства каналов превышает 1, реклама не окупается только для AdNonSense, FaceBoom и TipTop, самая лучшая окупаемость у lambdaMediaAds, MediaTornado и YRabbit.
# 
# Лучшее значение ROI показал канал YRabbit в середине июля, достигнув значения 7. Высокие показатели ROI дл YRabbit обусловлены в большей степени низкой стоимостью рекламы, чем высоким LTV.
# 
# Лучшие показатели конверсии у Mac, до 8 %. Худшие у PC, до 6,5 %.
# 
# В динамике конверсии 14-го заметен резкий провал у пользователей РС в конце июля до 5%, резкий скачок у пользователей andriod в 10-х числах сентября до 10,5% и резкий скачок у пользователей Mac в 20-х числах сентября до 10,5%.
# 
# Лучшие показатели конверсии у США, до 9,5%. Для остальных стран показатель примерно на одном уровне, около 5%.
# 
# В динамике конверсии 14-го для заметно снижение конверсии у Англии, Франции и Германии с 6% в мае до 5% в октябре, зато в США конверсия возрастала с 8% до 11% до середины июня, и сохраняется на уровне окло 9% на протяжении всего периода исследования.
# 
# Самая высокая конверсия у канала FaceBoom, к концу второй недели она достигает 12%, далее идут AdNonSense, lambdaMediaAds и TipTop, из конверсия 14-го дня примерно на уровне 10%.
# 
# Динамика конверсии пользователей на 14-й день наиболее нестабильна для lambdaMediaAds, в середине июня и августа она достигает 16%, а в начале сентября снижается до 5%.
# 
# Удержание платящих пользователей примерно одинаково для всех устройств и к 14 дню достигает 10% для iphohe, 12-13% для mac и android и 15% для PC.
# 
# Удержание неплатящих пользователей примерно одинаково для всех устройств и стремится к нулю.
# 
# В середине мая удержание платящих пользователей РС достигало 37%, так же неплохо себя показали РС и mac в конце мая и середине июля, тогда показатели удержания у обоих устройств достигали 25%.
# 
# Удержание платящих пользователей примерно одинаково для стран Европы и к 14 дню достигает 15-20%. Удержание для пользователей из США к 14 дню достигает лишь 10%
# 
# Удержание неплатящих пользователей примерно одинаково для всех стран и стремится к нулю.
# 
# Удержание платящих пользователей примерно одинаково для большинства каналов и к 14 дню достигает 20-25%. У MediaTomado и YRabbiit удержание 14 для достигает 10-12%. Хуже всего показывают себя AdNonSense и FaceBoom, их удержание близко к нулю.
# 
# Удержание неплатящих пользователей примерно одинаково для всех каналов и стремится к нулю.
# 
# В США работают 5 рекламных платформ, три из них RocketSuperAds, MediaTornado и YRabbit хорошо окупаются, но TipTop и FaceBoom не окупаются из-за высокой стоимости привлечения пользователей.
# 
# Во Франции, Германии и Англии работают 5 рекламных платформ, две из них LeapBob и lambdaMediaAds хорошо окупаются, еще две OppleCreativeMedia и WahooNetBanner окупаются немного хуже, и AdNonSense не окупается из-за высокой стоимости привлечения пользователей.
# 
# Для стран Европы эффективные рекламные каналы одни и те же, это LeapBob и lambdaMediaAds. LeapBob привлекает пользователей с невысокой пожизненной ценность, но затраты на этот рекламный канал очень малы, поэтому он хорошо окупается. lambdaMediaAds стоит дороже, но и привлекает более "качественных" пользователей.  AdNonSense не окупился ни в одной из стран Европы, от него следует отказаться.
# 
# Для США хорошую окупаемость показали RocketSuperAds, MediaTornado и YRabbit. Пожизненная ценность пользователей из этих каналов ниже чем у пользователей из TipTop, но при этом сама реклама на этих каналах стоит гораздо меньше.
# 
# В целом реклама перестала окупаться начиая с июня.
# 
# Показатель ROI к 14 дню лоходит до отметки 0,8, это значит, что к 14 дню возвращается только 80% стоимости рекламы.
# 
# Негативное воздействие на окупаемость рекламы могут оказывать:
# - Устройства:
#     - mac и iphone, их ROI ниже всего, при том что доли платящих пользователей для этих устройств выше других
# - Страны:
#     - США, показатель ROI для этой страны самый низкий. Реклама в США перестала окупаться в конце мая, и с тех пор показатель ROI не превышает единицу и стабильно снижается из-за огромных затрат на рекламу в TipTop и высоких затрат на рекламу в FaceBoom. Усугубляет ситуацию, что доля платящих пользователей из США составляет 78% от общего числа платящих пользователей. 
# - Рекламные каналы:
#     - Каналы FaceBoom и TipTop (США). Несмотря на то, что эти два канала привлекают 60% платящих пользователей, их окупаемость находится на уровне 54% для TipTop и 74% для FaceBoom 
#     - Канал AdNonSense (ЕС)
#     
# Проблемы окупаемости рекламы могут быть вызваны:
# 
# - Высокой стоимостью привлечения пользователей mac и iphone
# - Низкой конверсией пользователей РС
# - Низким удержанием пользователей из США
# - низким удержанием пользователей пришедших с каналов AdNonSense и FaceBoom
# - Низкой пожизненной стоимостью пользователей из Германии, Англии и Франции
# - Очень высокой стоимостью привлечения пользователей из США по каналу TipTop
# - Высокой стоимостью привлечения пользователей из ЕС по каналу AdNonSense
# - Низкой пожизненной стоимостью пользователей из каналов: OppleCreativeMedia, LeapBob, MediaTornado, YRabbit, AdNonSense, FaceBoom

# ### Напишите выводы <a id="shag6"></a>  
# [К содержанию](#soder)

# В таблице visits содержится информация о посещениях сайта. Таблица состоит из 6 столбцов и 309901 строк.
# 
# Структура visits:
# 
# User Id — уникальный идентификатор пользователя (целочисленный тип данных),
# 
# Region — страна пользователя (объектный тип данных),
# 
# Device — тип устройства пользователя (объектный тип данных),
# 
# Channel — идентификатор источника перехода (объектный тип данных),
# 
# Session Start — дата и время начала сессии (объектный тип данных),
# 
# Session End — дата и время окончания сессии (объектный тип данных).
# 
# Названия столбцов приведены к нижнему регистру.
# 
# Столбцы с датой и временем начала и окончания сессии приведены к типу datetime.
# 
# Пропусков, явных и неявных дубликатов не обнаружено.
# 
# В таблице orders содержится информация о заказах. Таблица состоит из 3 столбцов и 340212 строк.
# 
# Структура orders:
# 
# User Id — уникальный идентификатор пользователя (целочисленный тип данных),
# 
# Event Dt — дата и время покупки (объектный тип данных),
# 
# Revenue — сумма заказа (вещественный тип данных).
# 
# Названия столбцов приведены к нижнему регистру.
# 
# Столбец с датой и временем покупки приведены к типу datetime.
# 
# Пропусков и явных дубликатов не обнаружено.
# 
# В таблице costs содержится информация о расходах на рекламу. Таблица состоит из 3 столбцов и 1800 строк.
# 
# Структура costs: dt — дата проведения рекламной кампании (объектный тип данных),
# 
# Channel — идентификатор рекламного источника (объектный тип данных),
# 
# costs — расходы на эту кампанию (вещественный тип данных).
# 
# Названия столбцов приведены к нижнему регистру.
# 
# Столбец с датой проведения рекламной кампании приведены к типу datetime.
# 
# Пропусков, явных и неявных дубликатов не обнаружено.
# 
# Заданы функции для вычисления значений метрик:
# 
# - get_profiles() — для создания профилей пользователей
# - get_retention() — для подсчёта Retention Rate
# - get_conversion() — для подсчёта конверсии
# - get_ltv() — для подсчёта LTV
# 
# А также функции для построения графиков:
# 
# - filter_data() — для сглаживания данных,
# - plot_retention() — для построения графика Retention Rate,
# - plot_conversion() — для построения графика конверсии,
# - plot_ltv_roi — для визуализации LTV и ROI.
# 
# В таблице profiles содержится информация о 150 008 профилей пользователей, с датами первого посещения сайта, страной, каналом и устройством входа, в последных столбцах выведен месяц первого вхада и информация о том, покупал ли пользователь что-либо в течение исследуемого периода.
# 
# Минимальная дата привлечения пользователей: 2019-05-01.
# 
# Максимальная дата привлечения пользователей: 2019-10-27.
# 
# Они соответствуют границе исследуемого периода.
# 
# Больше всего зарегистрированных пользователей приходится на США, далее Франция, Англия и Германия.
# 
# Процент платящих пользователей больше всего также в США.
# 
# Отношение платящих пользователей к их общему количеству больше всего в США, далее Германия, Англия и Франция.
# 
# Пользователей из Германии меньше всего, но при этом их "качество" выше чем у пользователей из Франции и Англии.
# 
# Больше всего зарегистрированных пользователей заходят с Айфонов, далее Андроид, ПК и Мак.
# 
# Процент платящих пользователей больше всего также на Айфонах.
# 
# Отношение платящих пользователей к их общему количеству больше всего на Мак, далее Айфон, Андроид и ПК.
# 
# Пользователей Мак меньше всего, но при этом их "качество" выше чем у всех остальных пользователей.
# 
# Больше всего зарегистрированных пользователей (кроме тех кто приходят сами) приходят из FaceBoom их около 19%, далее идет TipTop (13%) и другие.
# 
# Процент платящих пользователей больше всего от каналов FaceBoom и TipTop.
# 
# Отношение платящих пользователей к их общему количеству больше всего на FaceBoom, далее AdNonSense, lambdaMediaAds и TipTop.
# 
# Пользователей AdNonSense и lambdaMediaAds меньше всего, но при этом их "качество" выше чем у всех остальных пользователей (кроме FaceBoom).
# 
# Всего за исследуемый период на рекламу было потрачено 105497,3$.
# 
# Больше всего потратили на рекламу в TipTop и FaceBoom.
# 
# Меньше всего на YRabbit, MediaTornado и lambdaMediaAds.
# 
# Расходы на рекламу в TipTop и FaceBoom постоянно растут, пики о обоих пришлись на 39 неделю года.
# 
# Расходы на остальные каналы привлечения пользователей примерно одинаковы, по ним не надлюдается значительных колебаний.
# 
# Расходы на рекламу в TipTop и FaceBoom постоянно растут, пик стоимости рекламы в TipTop пришелся на сентябрь.
# 
# Расходы на остальные каналы привлечения пользователей примерно одинаковы, по ним не наблюдается значительных колебаний.
# 
# В среднем самым дорогим источником привлечения пользователей является TipTop, средняя стоимость привлечения пользователей около 2,8$.
# 
# Стоимость привлечения постоянно растет, начиная с 1$ в мае до 3,5$ в сентябре-октябре. График динамики САС внешне напоминает лестницу, примерно в середине мая, двадцатых числах июня, конце июля и начале сентября наблюдаются резкие скачки стоимости привлечения пользователей, этому может быть два объяснения: в эти даты проводили рекламные кампании, или по каким-то причинам снижалось количество привлеченных клиентов.
# 
# Проверим это далее.
# 
# Привлечение клиентов через FaceBoom и AdNonSense обходится примерно в 1$, эти графики выглядят более стабильными.
# 
# Снижения количества привлеченных пользователей не наблюдается, скачки стоимости привлечения пользователей из TipTop обусловлены проведением ежемесячных рекламных кампаний.
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,9\\$, LTV колеблется от 0,7\\$ до 1,1\\$ на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# При этом стоимость привлечения пользователей (САС) постоянно увеличивается с 0,7\\$ в мае до 1,3\\$ к концу октября.
# 
# Коэффициент рентабельности инвестиций меньше единицы, значит реклама не окупается, так же на графике динамики ROI наблидается падение с 1,4 в мае до 0,6 к концу октября, это говорит о снижении эффективности рекламной стратегии.
# 
# Конверсия к 14 лайфтайму составляет около 8%.
# 
# В середине мая, конце июня, конце июля и конце августа конверсия четырнадцатого дня снижается до 7%.
# 
# В середине июня конверсия 14-го дня достигла пика 9,5%.
# 
# в сентябре-октябре колебания конверсии стали менее заметны.
# 
# Удержание платящих пользователей к 14-му дню составляет окло 10%, для неплатящих этот параметр составляет 10% на 1-2 день, а к 14-му приближается к нулю.
# 
# Удержание платящих пользователей на 14-й день в середине мая было ниже всего и составляла окло 6%, также падения до 7,5% наблюдались в конце июня и конце июля.
# 
# Самое высокое значение удержания платяжих пользователей на 14-й день наблюдалось в конце мая - начале июня, так же значение удержания превысио 15% в середине августа.
# 
# Удержание неплатящих пользвателей стабильно держится на уровне примерно 1% на протяжении всего исследуемого периода.
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,9\\$ для пользователей android, mac и iphone, а для пользователей РС немного ниже, на уровне 0,85\\$, LTV колеблется от 0,6\\$ до 1\\$ для пользователей РС, пики LTV для пользователей android, mac и iphone колеблется от 0,7\\$ до 1,1\\$ на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# Стоимость привлечения пользователей (САС) ниже всего для пользователей РС, она постоянно увеличивается с 0,7\\$ в мае до 0,9\\$ к концу октября. Далее идут пользователи android (с 0,7\\$ в мае до 1\\$ к середине сентября, а далее произошло небольшое снижение к концу октября). Пользователи mac и iphone "стоят" дороже всего (с 0,8\\$ в мае до 1,5\\$ к концу октября).
#  
# Коэффициент рентабельности инвестиций для пользователей android, mac и iphone меньше единицы, значит реклама не окупается, а для пользователей РС после 11 лайфтайма становится немного ниже единицы и начинает окупаться.
# 
# Так же на графике динамики ROI для пользователей android, mac и iphone наблидается падение с 1,2 в мае до 0,6 к концу октября. У пользователей РС до сентября значения ROI почти всегда превышало 1, но в сентябре и октябре снизилось до 0,7, это говорит о снижении эффективности рекламной стратегии.
# 
# Пожизненная ценность клиента к концу второй недели составляет около 0,8\\$ для пользователей из Германии, Англии и Франции, а для пользователей из США немного выше, и доходит до 1\\$, LTV у Американцев выше других на протяжении всего периода исследования, не наблюдается тенденции к понижению или повышению LTV за период.
# 
# Стоимость привлечения пользователей (САС) в июне снизилась для пользователей из Германии, Англии и Франции с 0,6\\$ до 0,4\\$, а для пользователей из США она постоянно повышалась до 1,8\\$.
# 
# Коэффициент рентабельности инвестиций для пользователей Германии, Англии и Франции превышает единицу начиная с 4-5 дня, а для Американцев доходит только до 0,7, значит реклама в США не окупается.
# 
# Так же на графике динамики ROI для пользователей из Германии, Англии и Франции ROI стабильно выше 1, а для пользователей из Англии в середине июля и августа доходит до 2,25 и 2,6 соответственно. У пользователей из США значения ROI с мая до конца октября упали с 1,2 до 0,5 это говорит о снижении эффективности рекламной стратегии для США.
# 
# LTV выше всего для пользователей lambdaMediaAds и TipTop и к концу втрой недели доходит до 1,5\\$ и 1,8\\$ соответственно.
# 
# Самые лучшие показатели LTV в динамике также у lambdaMediaAds и TipTop, причем lambdaMediaAds с начала сентября стабильно растет, а в начале июня LTV lambdaMediaAds достигал 3,2\\$.
# 
# Стоимость привлечения пользователей стабильна для всех каналов кроме TipTop, для него стоимость привлечения возрасла с 1\\$ до 3,5\\$.
# 
# Показатель ROI для большинства каналов превышает 1, реклама не окупается только для AdNonSense, FaceBoom и TipTop, самая лучшая окупаемость у lambdaMediaAds, MediaTornado и YRabbit.
# 
# Лучшее значение ROI показал канал YRabbit в середине июля, достигнув значения 7. Высокие показатели ROI дл YRabbit обусловлены в большей степени низкой стоимостью рекламы, чем высоким LTV.
# 
# Лучшие показатели конверсии у Mac, до 8 %. Худшие у PC, до 6,5 %.
# 
# В динамике конверсии 14-го заметен резкий провал у пользователей РС в конце июля до 5%, резкий скачок у пользователей andriod в 10-х числах сентября до 10,5% и резкий скачок у пользователей Mac в 20-х числах сентября до 10,5%.
# 
# Лучшие показатели конверсии у США, до 9,5%. Для остальных стран показатель примерно на одном уровне, около 5%.
# 
# В динамике конверсии 14-го для заметно снижение конверсии у Англии, Франции и Германии с 6% в мае до 5% в октябре, зато в США конверсия возрастала с 8% до 11% до середины июня, и сохраняется на уровне окло 9% на протяжении всего периода исследования.
# 
# Самая высокая конверсия у канала FaceBoom, к концу второй недели она достигает 12%, далее идут AdNonSense, lambdaMediaAds и TipTop, из конверсия 14-го дня примерно на уровне 10%.
# 
# Динамика конверсии пользователей на 14-й день наиболее нестабильна для lambdaMediaAds, в середине июня и августа она достигает 16%, а в начале сентября снижается до 5%.
# 
# Удержание платящих пользователей примерно одинаково для всех устройств и к 14 дню достигает 10% для iphohe, 12-13% для mac и android и 15% для PC.
# 
# Удержание неплатящих пользователей примерно одинаково для всех устройств и стремится к нулю.
# 
# В середине мая удержание платящих пользователей РС достигало 37%, так же неплохо себя показали РС и mac в конце мая и середине июля, тогда показатели удержания у обоих устройств достигали 25%.
# 
# Удержание платящих пользователей примерно одинаково для стран Европы и к 14 дню достигает 15-20%. Удержание для пользователей из США к 14 дню достигает лишь 10%
# 
# Удержание неплатящих пользователей примерно одинаково для всех стран и стремится к нулю.
# 
# Удержание платящих пользователей примерно одинаково для большинства каналов и к 14 дню достигает 20-25%. У MediaTomado и YRabbiit удержание 14 для достигает 10-12%. Хуже всего показывают себя AdNonSense и FaceBoom, их удержание близко к нулю.
# 
# Удержание неплатящих пользователей примерно одинаково для всех каналов и стремится к нулю.
# 
# В США работают 5 рекламных платформ, три из них RocketSuperAds, MediaTornado и YRabbit хорошо окупаются, но TipTop и FaceBoom не окупаются из-за высокой стоимости привлечения пользователей.
# 
# Во Франции, Германии и Англии работают 5 рекламных платформ, две из них LeapBob и lambdaMediaAds хорошо окупаются, еще две OppleCreativeMedia и WahooNetBanner окупаются немного хуже, и AdNonSense не окупается из-за высокой стоимости привлечения пользователей.
# 
# Для стран Европы эффективные рекламные каналы одни и те же, это LeapBob и lambdaMediaAds. LeapBob привлекает пользователей с невысокой пожизненной ценность, но затраты на этот рекламный канал очень малы, поэтому он хорошо окупается. lambdaMediaAds стоит дороже, но и привлекает более "качественных" пользователей.  AdNonSense не окупился ни в одной из стран Европы, от него следует отказаться.
# 
# Для США хорошую окупаемость показали RocketSuperAds, MediaTornado и YRabbit. Пожизненная ценность пользователей из этих каналов ниже чем у пользователей из TipTop, но при этом сама реклама на этих каналах стоит гораздо меньше.
# 
# В целом реклама перестала окупаться начиая с июня.
# 
# Показатель ROI к 14 дню лоходит до отметки 0,8, это значит, что к 14 дню возвращается только 80% стоимости рекламы.
# 
# Негативное воздействие на окупаемость рекламы могут оказывать:
# - Устройства:
#     - mac и iphone, их ROI ниже всего, при том что доли платящих пользователей для этих устройств выше других
# - Страны:
#     - США, показатель ROI для этой страны самый низкий. Реклама в США перестала окупаться в конце мая, и с тех пор показатель ROI не превышает единицу и стабильно снижается из-за огромных затрат на рекламу в TipTop и высоких затрат на рекламу в FaceBoom. Усугубляет ситуацию, что доля платящих пользователей из США составляет 78% от общего числа платящих пользователей. 
# - Рекламные каналы:
#     - Каналы FaceBoom и TipTop (США). Несмотря на то, что эти два канала привлекают 60% платящих пользователей, их окупаемость находится на уровне 54% для TipTop и 74% для FaceBoom 
#     - Канал AdNonSense (ЕС)
#     
# Проблемы окупаемости рекламы могут быть вызваны:
# 
# - Высокой стоимостью привлечения пользователей mac и iphone
# - Низкой конверсией пользователей РС
# - Низким удержанием пользователей из США
# - низким удержанием пользователей пришедших с каналов AdNonSense и FaceBoom
# - Низкой пожизненной стоимостью пользователей из Германии, Англии и Франции
# - Очень высокой стоимостью привлечения пользователей из США по каналу TipTop
# - Высокой стоимостью привлечения пользователей из ЕС по каналу AdNonSense
# - Низкой пожизненной стоимостью пользователей из каналов: OppleCreativeMedia, LeapBob, MediaTornado, YRabbit, AdNonSense, FaceBoom
# 
# 
# На основании проведенного анализа, можно составить следующие рекомендации для отдела маркетинга:
# - Пересмотреть эффективность каналов привлечения пользователей из разных стран.
#     - снизить затраты на рекламу в TipTop и FaceBoom в США, развивать рекламу в США стоит на каналах RocketSuperAds, MediaTornado и YRabbit
#     - снизить затраты на рекламу в AdNonSense в ЕС, 
# - Обратить внимание на конверсию пользователей РС
# Возможно пользователи РС сталкиваются с техническими проблемами
# - Пересмотреть эффективность рекламы для пользователей mac и iphone
# Стоимость их привлечения как и доли платящих пользователей для этих устройств выше других. Однако пожизенная ценность этих пользователей не отличается от остальных
# - Обратить внимание на "качество" целевой аудитории
# Доля пользователей из США  составляет 78%, при этом у них самый низкий показатель ROI. Это связано с огромными затратами на рекламу в США, которые не окупаются. При этом затраты на рекламу в остальных странах окупаются уже через неделю. Необходимо сниизть затраты на рекламу в США и привлечь больше пользователей из других стран, при этом повысив их конверсию.
# - Повысить конверсию органических пользователей
# При доле органич
