-- 1. Найдите количество вопросов, которые набрали больше 300 очков или как минимум 100 раз были добавлены в «Закладки».

SELECT COUNT(p.id)
FROM stackoverflow.posts p
JOIN stackoverflow.post_types pt ON p.post_type_id = pt.id
WHERE pt.type = 'Question' AND (p.score > 300 OR p.favorites_count >= 100)

-- 2. Сколько в среднем в день задавали вопросов с 1 по 18 ноября 2008 включительно? Результат округлите до целого числа.

WITH c AS 
(SELECT COUNT(*) AS c_id
FROM stackoverflow.posts p
JOIN stackoverflow.post_types pt ON p.post_type_id = pt.id
WHERE pt.type = 'Question' AND DATE_TRUNC('day', creation_date)::date BETWEEN '2008-11-01' AND '2008-11-18'
GROUP BY DATE_TRUNC('day', creation_date))

SELECT ROUND(AVG(c_id))
FROM c

-- 3.Сколько пользователей получили значки сразу в день регистрации? Выведите количество уникальных пользователей.

SELECT COUNT(DISTINCT u.id)
FROM stackoverflow.users u 
JOIN stackoverflow.badges b ON u.id =b.user_id
WHERE DATE_TRUNC('day', u.creation_date) = DATE_TRUNC('day', b.creation_date)

-- 4.Сколько уникальных постов пользователя с именем Joel Coehoorn получили хотя бы один голос?

SELECT COUNT (DISTINCT p.id)
FROM stackoverflow.users u
JOIN stackoverflow.posts p ON u.id=p.user_id
JOIN stackoverflow.votes v ON p.id = v.post_id
WHERE u.display_name = 'Joel Coehoorn' 
AND v.vote_type_id != 0

-- 5.Выгрузите все поля таблицы vote_types. Добавьте к таблице поле rank, в которое войдут номера записей в обратном порядке. 
-- Таблица должна быть отсортирована по полю id.

SELECT *,
       ROW_NUMBER() OVER (ORDER BY id DESC) AS rank
FROM stackoverflow.vote_types
ORDER BY id

-- 6. Отберите 10 пользователей, которые поставили больше всего голосов типа Close. 
--Отобразите таблицу из двух полей: идентификатором пользователя и количеством голосов. 
--Отсортируйте данные сначала по убыванию количества голосов, потом по убыванию значения идентификатора пользователя.

SELECT u.id,
       COUNT(v.id) AS cnt
FROM stackoverflow.users u
JOIN stackoverflow.votes v ON u.id= v.user_id
JOIN stackoverflow.vote_types vt ON v.vote_type_id=vt.id
WHERE vt.name = 'Close'
GROUP BY u.id
ORDER BY cnt DESC, u.id DESC
LIMIT 10

-- 7. Отберите 10 пользователей по количеству значков, полученных в период с 15 ноября по 15 декабря 2008 года включительно.
--Отобразите несколько полей:
--идентификатор пользователя;
--число значков;
--место в рейтинге — чем больше значков, тем выше рейтинг.
--Пользователям, которые набрали одинаковое количество значков, присвойте одно и то же место в рейтинге.
--Отсортируйте записи по количеству значков по убыванию, а затем по возрастанию значения идентификатора пользователя.

SELECT user_id,
       COUNT(*),
       DENSE_RANK() OVER(ORDER BY COUNT(*) DESC)
FROM stackoverflow.badges
WHERE DATE_TRUNC('day', creation_date) BETWEEN '2008-11-15' AND '2008-12-15'
GROUP BY user_id
ORDER BY COUNT(*) DESC, user_id
LIMIT 10

-- 8. Сколько в среднем очков получает пост каждого пользователя?
--Сформируйте таблицу из следующих полей:
--заголовок поста;
--идентификатор пользователя;
--число очков поста;
--среднее число очков пользователя за пост, округлённое до целого числа.
--Не учитывайте посты без заголовка, а также те, что набрали ноль очков.

SELECT title,
       user_id,
       score,
       ROUND(AVG(score) OVER (PARTITION BY user_id))
FROM stackoverflow.posts
WHERE title != ' ' AND score != 0

--9. Отобразите заголовки постов, которые были написаны пользователями, получившими более 1000 значков. Посты без заголовков не должны попасть в список.

WITH t_1 AS 
(SELECT b.user_id
FROM stackoverflow.badges b
GROUP BY b.user_id
HAVING COUNT(b.id) > 1000)


SELECT p.title
FROM stackoverflow.posts p
JOIN t_1 ON p.user_id = t_1.user_id
WHERE p.title != ' '

-- 10. Напишите запрос, который выгрузит данные о пользователях из Канады (англ. Canada). 
--Разделите пользователей на три группы в зависимости от количества просмотров их профилей:
--пользователям с числом просмотров больше либо равным 350 присвойте группу 1;
--пользователям с числом просмотров меньше 350, но больше либо равно 100 — группу 2;
--пользователям с числом просмотров меньше 100 — группу 3.
--Отобразите в итоговой таблице идентификатор пользователя, количество просмотров профиля и группу. 
--Пользователи с количеством просмотров меньше либо равным нулю не должны войти в итоговую таблицу.

SELECT id,
       views,
       CASE 
           WHEN views >= 350 THEN 1
           WHEN views < 350 AND views >= 100 THEN 2
           WHEN views < 100 THEN 3
       END
FROM stackoverflow.users
WHERE views != 0 
AND location LIKE '%Canada%'

-- 11.Дополните предыдущий запрос. Отобразите лидеров каждой группы — пользователей, которые набрали максимальное число просмотров в своей группе. 
--Выведите поля с идентификатором пользователя, группой и количеством просмотров. 
--Отсортируйте таблицу по убыванию просмотров, а затем по возрастанию значения идентификатора.

WITH t_1 AS 
(SELECT id,
       views,
       (CASE 
           WHEN views >= 350 THEN 1
           WHEN views < 350 AND views >= 100 THEN 2
           WHEN views < 100 THEN 3
       END) AS cat
FROM stackoverflow.users
WHERE views != 0 
AND location LIKE '%Canada%'),

t_2 AS 
(SELECT MAX(views) AS max
FROM t_1
GROUP BY cat)

SELECT t_1.id,
       t_1.cat, 
       t_2.max
FROM t_2 LEFT JOIN t_1 ON t_2.max = t_1.views 
ORDER BY max DESC, id

-- 12. Посчитайте ежедневный прирост новых пользователей в ноябре 2008 года. Сформируйте таблицу с полями:
--номер дня;
--число пользователей, зарегистрированных в этот день;
--сумму пользователей с накоплением.

SELECT EXTRACT(DAY FROM DATE_TRUNC('day', creation_date)) AS day,
       COUNT (id) AS cnt,
       SUM(COUNT (id)) OVER(ORDER BY EXTRACT(DAY FROM DATE_TRUNC('day', creation_date)))
FROM stackoverflow.users
WHERE DATE_TRUNC('day', creation_date) BETWEEN '2008-11-01' AND '2008-11-30'
GROUP BY EXTRACT(DAY FROM DATE_TRUNC('day', creation_date))

-- 13. Для каждого пользователя, который написал хотя бы один пост, найдите интервал между регистрацией и временем создания первого поста. Отобразите:
--идентификатор пользователя;
--разницу во времени между регистрацией и первым постом.

WITH regs AS 
(SELECT id,
        creation_date
FROM stackoverflow.users),

fp AS 
(SELECT user_id,
        MIN(creation_date) AS fp
FROM stackoverflow.posts
GROUP BY user_id)

SELECT id, 
       fp - creation_date
FROM regs JOIN fp ON regs.id = fp.user_id

-- 14. Выведите общую сумму просмотров у постов, опубликованных в каждый месяц 2008 года. Если данных за какой-либо месяц в базе нет, такой месяц можно пропустить. 
--Результат отсортируйте по убыванию общего количества просмотров.

SELECT DATE_TRUNC('month', creation_date)::date,
       SUM(views_count)
FROM stackoverflow.posts
WHERE EXTRACT (YEAR FROM creation_date) = '2008'
GROUP BY DATE_TRUNC('month', creation_date)
ORDER BY SUM(views_count) DESC

-- 15. Выведите имена самых активных пользователей, которые в первый месяц после регистрации (включая день регистрации) дали больше 100 ответов. 
--Вопросы, которые задавали пользователи, не учитывайте. Для каждого имени пользователя выведите количество уникальных значений user_id. 
--Отсортируйте результат по полю с именами в лексикографическом порядке.

SELECT u.display_name,
       COUNT(DISTINCT u.id)
FROM stackoverflow.users u 
JOIN stackoverflow.posts p ON u.id = p.user_id
JOIN stackoverflow.post_types pt ON p.post_type_id = pt.id
WHERE pt.type = 'Answer' 
AND (DATE_TRUNC('day', p.creation_date) >= DATE_TRUNC('day', u.creation_date))
AND  (DATE_TRUNC('day', p.creation_date) <= DATE_TRUNC('day', u.creation_date)+ INTERVAL '1 month')
GROUP BY u.display_name
HAVING COUNT(p.id) > 100
ORDER BY u.display_name

-- 16. Выведите количество постов за 2008 год по месяцам. Отберите посты от пользователей, 
--которые зарегистрировались в сентябре 2008 года и сделали хотя бы один пост в декабре того же года. 
--Отсортируйте таблицу по значению месяца по убыванию.

WITH t_1 AS 
(SELECT u.id AS user_id,
       DATE_TRUNC('month', u.creation_date) AS user_creation_date,
       p.id AS post_id,
       DATE_TRUNC('month', p.creation_date) AS post_creation_date
FROM stackoverflow.users u JOIN stackoverflow.posts p ON u.id = p.user_id
WHERE DATE_TRUNC('month', u.creation_date)  = '2008-09-01' AND DATE_TRUNC('month', p.creation_date) = '2008-12-01')

SELECT DATE_TRUNC ('month', p.creation_date)::date,
       COUNT(DISTINCT p.id)
FROM t_1 JOIN stackoverflow.posts p ON t_1.user_id = p.user_id
GROUP BY DATE_TRUNC ('month', p.creation_date)
ORDER BY DATE_TRUNC ('month', p.creation_date) DESC

-- 17. Используя данные о постах, выведите несколько полей:
--идентификатор пользователя, который написал пост;
--дата создания поста;
--количество просмотров у текущего поста;
--сумма просмотров постов автора с накоплением.
--Данные в таблице должны быть отсортированы по возрастанию идентификаторов пользователей, а данные об одном и том же пользователе — по возрастанию даты создания поста.

SELECT user_id,
       creation_date,
       views_count,
       SUM(views_count) OVER (PARTITION BY user_id ORDER BY creation_date)
FROM stackoverflow.posts

-- 18. Сколько в среднем дней в период с 1 по 7 декабря 2008 года включительно пользователи взаимодействовали с платформой? 
--Для каждого пользователя отберите дни, в которые он или она опубликовали хотя бы один пост. 
--Нужно получить одно целое число — не забудьте округлить результат.

WITH t_1 AS 
(SELECT user_id,
       COUNT(DISTINCT DATE_TRUNC('day', creation_date)) AS cnt
FROM stackoverflow.posts
WHERE DATE_TRUNC('day', creation_date) BETWEEN '2008-12-01' AND '2008-12-07'
GROUP BY user_id)

SELECT ROUND(AVG(cnt))
FROM t_1

-- 19. На сколько процентов менялось количество постов ежемесячно с 1 сентября по 31 декабря 2008 года? Отобразите таблицу со следующими полями:
--Номер месяца.
--Количество постов за месяц.
--Процент, который показывает, насколько изменилось количество постов в текущем месяце по сравнению с предыдущим.
--Если постов стало меньше, значение процента должно быть отрицательным, если больше — положительным. Округлите значение процента до двух знаков после запятой.
--Напомним, что при делении одного целого числа на другое в PostgreSQL в результате получится целое число, округлённое до ближайшего целого вниз. 
--Чтобы этого избежать, переведите делимое в тип numeric.

WITH t_1 AS 
(SELECT EXTRACT (MONTH FROM creation_date) AS month,
       COUNT (DISTINCT id) AS posts_cnt
FROM stackoverflow.posts
WHERE EXTRACT (MONTH FROM creation_date) >= 9
GROUP BY EXTRACT (MONTH FROM creation_date)
ORDER BY EXTRACT (MONTH FROM creation_date))

SELECT *,
       ROUND(((posts_cnt::numeric / LAG (posts_cnt) OVER (ORDER BY month)) - 1)*100, 2)
FROM t_1       


-- 20. Найдите пользователя, который опубликовал больше всего постов за всё время с момента регистрации. 
--Выведите данные его активности за октябрь 2008 года в таком виде:
--номер недели;
--дата и время последнего поста, опубликованного на этой неделе.

WITH leader_id AS 
(SELECT user_id,
       COUNT(DISTINCT id) AS posts_cnt
FROM stackoverflow.posts
GROUP BY user_id
ORDER BY posts_cnt DESC
LIMIT 1),

weeks AS 
(SELECT id,
       creation_date,
       EXTRACT(WEEK FROM creation_date) AS week
FROM stackoverflow.posts p JOIN leader_id ON p.user_id = leader_id.user_id
WHERE EXTRACT(MONTH FROM creation_date) = 10)

SELECT DISTINCT week, 
       MAX(creation_date) OVER (PARTITION BY week)
FROM  weeks   
