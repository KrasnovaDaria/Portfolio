-- 1. Отобразите все записи из таблицы company по компаниям, которые закрылись.

SELECT *
FROM company
WHERE status = 'closed';

-- 2. Отобразите количество привлечённых средств для новостных компаний США. 
-- Используйте данные из таблицы company. Отсортируйте таблицу по убыванию значений в поле funding_total.

SELECT funding_total
FROM company
WHERE country_code = 'USA'
      AND category_code = 'news'
ORDER BY funding_total DESC;

-- Найдите общую сумму сделок по покупке одних компаний другими в долларах. 
-- Отберите сделки, которые осуществлялись только за наличные с 2011 по 2013 год включительно.

SELECT SUM(price_amount)
FROM acquisition
WHERE term_code = 'cash'
      AND EXTRACT (YEAR FROM acquired_at) IN (2011, 2012, 2013)

-- 4. Отобразите имя, фамилию и названия аккаунтов людей в поле network_username, у которых названия аккаунтов начинаются на 'Silver'.

SELECT first_name,
       last_name,
       network_username
FROM people
WHERE network_username LIKE 'Silver%'

-- 5. Выведите на экран всю информацию о людях, у которых названия аккаунтов в поле network_username содержат подстроку 'money', а фамилия начинается на 'K'.

SELECT *
FROM people
WHERE network_username LIKE '%money%'
      AND last_name LIKE 'K%'

-- 6.Для каждой страны отобразите общую сумму привлечённых инвестиций, которые получили компании, зарегистрированные в этой стране. 
-- Страну, в которой зарегистрирована компания, можно определить по коду страны. Отсортируйте данные по убыванию суммы.
SELECT country_code,
       SUM(funding_total)
FROM company
GROUP BY country_code
ORDER BY SUM(funding_total) DESC

--7.Составьте таблицу, в которую войдёт дата проведения раунда, а также минимальное и максимальное значения суммы инвестиций, привлечённых в эту дату.
-- Оставьте в итоговой таблице только те записи, в которых минимальное значение суммы инвестиций не равно нулю и не равно максимальному значению.

SELECT funded_at,
       MIN(raised_amount),
       MAX(raised_amount)
FROM funding_round
GROUP BY funded_at
HAVING MIN(raised_amount) > 0 
       AND MIN(raised_amount) !=  MAX(raised_amount) 

-- 8.Создайте поле с категориями:
-- Для фондов, которые инвестируют в 100 и более компаний, назначьте категорию high_activity.
-- Для фондов, которые инвестируют в 20 и более компаний до 100, назначьте категорию middle_activity.
-- Если количество инвестируемых компаний фонда не достигает 20, назначьте категорию low_activity.
-- Отобразите все поля таблицы fund и новое поле с категориями.

SELECT *,
      CASE WHEN invested_companies >= 100 THEN 'high_activity'
           WHEN invested_companies < 100 AND invested_companies >= 20 THEN 'middle_activity'
           WHEN invested_companies < 20 THEN 'low_activity'
      END
FROM fund

-- 9. Для каждой из категорий, назначенных в предыдущем задании, 
--посчитайте округлённое до ближайшего целого числа среднее количество инвестиционных раундов, в которых фонд принимал участие. 
--Выведите на экран категории и среднее число инвестиционных раундов. Отсортируйте таблицу по возрастанию среднего.

SELECT CASE
           WHEN invested_companies>=100 THEN 'high_activity'
           WHEN invested_companies>=20 THEN 'middle_activity'
           ELSE 'low_activity'
       END AS activity,
       ROUND(AVG(investment_rounds))
FROM fund
GROUP BY activity
ORDER BY ROUND(AVG(investment_rounds))

-- 10. Проанализируйте, в каких странах находятся фонды, которые чаще всего инвестируют в стартапы. 
-- Для каждой страны посчитайте минимальное, максимальное и среднее число компаний, в которые инвестировали фонды этой страны, основанные с 2010 по 2012 год включительно. 
-- Исключите страны с фондами, у которых минимальное число компаний, получивших инвестиции, равно нулю. 
-- Выгрузите десять самых активных стран-инвесторов: отсортируйте таблицу по среднему количеству компаний от большего к меньшему. 
-- Затем добавьте сортировку по коду страны в лексикографическом порядке.

SELECT country_code,
       MIN(invested_companies) AS min,
       MAX(invested_companies) AS max,
       AVG(invested_companies) AS avg
FROM fund
WHERE EXTRACT (YEAR FROM founded_at) IN (2010, 2011, 2012)
GROUP BY country_code
HAVING MIN(invested_companies) != 0
ORDER BY avg DESC, country_code
LIMIT 10

-- 11.Отобразите имя и фамилию всех сотрудников стартапов. Добавьте поле с названием учебного заведения, которое окончил сотрудник, если эта информация известна.

SELECT p.first_name,
       p.last_name,
       e.instituition
FROM people AS p
LEFT OUTER JOIN education AS e ON e.person_id = p.id

-- 12. Для каждой компании найдите количество учебных заведений, которые окончили её сотрудники. 
-- Выведите название компании и число уникальных названий учебных заведений. 
-- Составьте топ-5 компаний по количеству университетов.

SELECT c.name,
       COUNT (DISTINCT e.instituition)
FROM company AS c
JOIN people AS p ON c.id = p.company_id
JOIN education AS e ON p.id = e.person_id
GROUP BY c.name
ORDER BY COUNT (DISTINCT e.instituition) DESC
LIMIT 5

-- 13.Составьте список с уникальными названиями закрытых компаний, для которых первый раунд финансирования оказался последним.

SELECT DISTINCT name
FROM company
JOIN funding_round ON company.id = funding_round.company_id
WHERE funding_round.company_id IN (SELECT company_id
       FROM funding_round
       WHERE is_first_round = 1 
       AND is_last_round = 1)
      AND company.status = 'closed'

--14.Составьте список уникальных номеров сотрудников, которые работают в компаниях, отобранных в предыдущем задании.

SELECT people.id
FROM people
JOIN company ON people.company_id = company.id
WHERE company.id IN
(SELECT DISTINCT company.id
FROM company
JOIN funding_round ON company.id = funding_round.company_id
WHERE funding_round.company_id IN (SELECT company_id
       FROM funding_round
       WHERE is_first_round = 1 
       AND is_last_round = 1)
      AND company.status = 'closed')

--15.Составьте таблицу, куда войдут уникальные пары с номерами сотрудников из предыдущей задачи и учебным заведением, которое окончил сотрудник.

SELECT DISTINCT people.id,
                education.instituition
FROM people
JOIN company ON people.company_id = company.id
JOIN education ON people.id = education.person_id
WHERE company.id IN
                   (SELECT DISTINCT company.id
                   FROM company
                   JOIN funding_round ON company.id = funding_round.company_id
                   WHERE funding_round.company_id IN (SELECT company_id
                                                      FROM funding_round
                                                      WHERE is_first_round = 1 
                                                      AND is_last_round = 1)
                   AND company.status = 'closed')

-- 16.Посчитайте количество учебных заведений для каждого сотрудника из предыдущего задания. 
--При подсчёте учитывайте, что некоторые сотрудники могли окончить одно и то же заведение дважды.

SELECT DISTINCT people.id,
                COUNT (education.instituition)
FROM people
JOIN company ON people.company_id = company.id
JOIN education ON people.id = education.person_id
WHERE company.id IN
                   (SELECT DISTINCT company.id
                   FROM company
                   JOIN funding_round ON company.id = funding_round.company_id
                   WHERE funding_round.company_id IN (SELECT company_id
                                                      FROM funding_round
                                                      WHERE is_first_round = 1 
                                                      AND is_last_round = 1)
                   AND company.status = 'closed')
GROUP BY people.id

-- 17.Дополните предыдущий запрос и выведите среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники разных компаний. 
-- Нужно вывести только одну запись, группировка здесь не понадобится.

SELECT AVG(count)
FROM
(SELECT DISTINCT people.id,
                COUNT (education.instituition) AS count
FROM people
JOIN company ON people.company_id = company.id
JOIN education ON people.id = education.person_id
WHERE company.id IN
                   (SELECT DISTINCT company.id
                   FROM company
                   JOIN funding_round ON company.id = funding_round.company_id
                   WHERE funding_round.company_id IN (SELECT company_id
                                                      FROM funding_round
                                                      WHERE is_first_round = 1 
                                                      AND is_last_round = 1)
                   AND company.status = 'closed')
GROUP BY people.id) AS t_1

--18.Напишите похожий запрос: выведите среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники Socialnet.

SELECT AVG(count)
FROM
(SELECT DISTINCT people.id,
                COUNT (education.instituition) AS count
FROM people
JOIN company ON people.company_id = company.id
JOIN education ON people.id = education.person_id
WHERE company.name = 'Socialnet'
GROUP BY people.id) AS t_1

-- 19.Составьте таблицу из полей:
--name_of_fund — название фонда;
--name_of_company — название компании;
--amount — сумма инвестиций, которую привлекла компания в раунде.
--В таблицу войдут данные о компаниях, в истории которых было больше шести важных этапов, а раунды финансирования проходили с 2012 по 2013 год включительно.

SELECT f.name AS name_of_fund,
       c.name AS name_of_company,
       fr.raised_amount AS amount
FROM investment AS i
JOIN company AS c ON i.company_id = c.id
JOIN fund AS f ON i.fund_id = f.id
JOIN funding_round AS fr ON i.funding_round_id = fr.id
WHERE EXTRACT (YEAR FROM funded_at) IN (2012, 2013)
AND c.name IN (SELECT c.name
              FROM company AS c
              JOIN funding_round AS fr ON fr.company_id = c.id
              WHERE c.milestones >6)

--20.Выгрузите таблицу, в которой будут такие поля:
--название компании-покупателя;
--сумма сделки;
--название компании, которую купили;
--сумма инвестиций, вложенных в купленную компанию;
--доля, которая отображает, во сколько раз сумма покупки превысила сумму вложенных в компанию инвестиций, округлённая до ближайшего целого числа.
--Не учитывайте те сделки, в которых сумма покупки равна нулю. Если сумма инвестиций в компанию равна нулю, исключите такую компанию из таблицы. 
--Отсортируйте таблицу по сумме сделки от большей к меньшей, а затем по названию купленной компании в лексикографическом порядке. 
--Ограничьте таблицу первыми десятью записями.

SELECT c_1.name AS acquiring_company,
       a.price_amount AS price_amount,
       c_2.name AS acquired_company,
       c_2.funding_total AS funding_total,
       ROUND(price_amount/c_2.funding_total) AS perc
FROM acquisition AS a
LEFT JOIN company AS c_1 ON a.acquiring_company_id  = c_1.id
LEFT JOIN company AS c_2 ON a.acquired_company_id  = c_2.id
WHERE a.price_amount > 0
AND c_2.funding_total > 0
ORDER BY price_amount DESC, acquired_company
LIMIT 10

-- 21.Выгрузите таблицу, в которую войдут названия компаний из категории social, получившие финансирование с 2010 по 2013 год включительно. 
-- Проверьте, что сумма инвестиций не равна нулю. Выведите также номер месяца, в котором проходил раунд финансирования.

SELECT c.name AS name,
       EXTRACT (MONTH FROM fr.funded_at) AS funded_at
FROM company AS c
JOIN funding_round AS fr ON c.id = fr.company_id
WHERE c.category_code = 'social'
      AND EXTRACT (YEAR FROM fr.funded_at) BETWEEN 2010 AND 2013
      AND fr.raised_amount > 0

--22. Отберите данные по месяцам с 2010 по 2013 год, когда проходили инвестиционные раунды. 
--Сгруппируйте данные по номеру месяца и получите таблицу, в которой будут поля:
--номер месяца, в котором проходили раунды;
--количество уникальных названий фондов из США, которые инвестировали в этом месяце;
--количество компаний, купленных за этот месяц;
--общая сумма сделок по покупкам в этом месяце.

WITH
t_1 AS 
(SELECT EXTRACT (MONTH FROM fr.funded_at) AS funding_round_month,
        COUNT (DISTINCT f.name) AS founds_count
FROM fund AS f
JOIN investment AS i ON f.id = i.fund_id
JOIN funding_round AS fr ON i.funding_round_id = fr.id
WHERE f.country_code = 'USA' 
AND EXTRACT (YEAR FROM fr.funded_at) BETWEEN 2010 AND 2013
GROUP BY funding_round_month),
t_2 AS        
(SELECT EXTRACT (MONTH FROM a.acquired_at) AS funding_round_month,
       COUNT(a.acquired_company_id) AS acquired_company_count,
       SUM (a.price_amount) AS trade_sum
FROM acquisition AS a
WHERE EXTRACT (YEAR FROM a.acquired_at) BETWEEN 2010 AND 2013
GROUP BY funding_round_month)
SELECT t_1.funding_round_month,
       t_1.founds_count,
       t_2.acquired_company_count,
       t_2.trade_sum
FROM t_1 JOIN t_2 ON t_1.funding_round_month = t_2.funding_round_month

-- 23. Составьте сводную таблицу и выведите среднюю сумму инвестиций для стран, в которых есть стартапы, зарегистрированные в 2011, 2012 и 2013 годах. 
--Данные за каждый год должны быть в отдельном поле. Отсортируйте таблицу по среднему значению инвестиций за 2011 год от большего к меньшему.

WITH
     inv_2011 AS 
     (SELECT country_code AS country_code,
             AVG (funding_total) AS res_2011
     FROM company
     WHERE EXTRACT (YEAR FROM founded_at) = 2011
     GROUP BY country_code),
     
     inv_2012 AS 
     (SELECT country_code AS country_code,
             AVG (funding_total) AS res_2012
     FROM company
     WHERE EXTRACT (YEAR FROM founded_at) = 2012
     GROUP BY country_code),
     
     inv_2013 AS 
     (SELECT country_code AS country_code,
             AVG (funding_total) AS res_2013
     FROM company
     WHERE EXTRACT (YEAR FROM founded_at) = 2013
     GROUP BY country_code)

SELECT inv_2011.country_code,
       inv_2011.res_2011,
       inv_2012.res_2012,
       inv_2013.res_2013
FROM inv_2011 
INNER JOIN inv_2012 ON inv_2011.country_code = inv_2012.country_code
INNER JOIN inv_2013 ON inv_2011.country_code = inv_2013.country_code

ORDER BY res_2011 DESC

