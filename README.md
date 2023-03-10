Для задачи создания рекомендательной системы книг подойдут гибридные подходы к рекомендательным системам, так как они будут обрабатывать наиболее полный портрет пользователя.
В основе своей используется два подхода - один основанный на связях между пользователями, другой на качествах объекта. 

Первый не подходит так как он не учитывает характеристик объекта типа жанров и авторов и будет давать очень генерализированные результаты по предпочтениям пользователей. 

Второй является более подходящим, но ,если в расчет будет браться еще и опыт других пользователей, результат будет лучше

Из гибридных моделей рекомендательных систем можно использовать NeuMF, так как она позволяет одновременно иметь во внимании атрибуты книги и пользователя. Основу ее архитектуры можно оставить.

Другой подходящей архитектурой является NCF, но она является родоначальницей NeuMF.

 ![image](https://user-images.githubusercontent.com/94608666/211184605-5058d944-b8b6-48a0-8ef6-d10540cba63e.png)

На вход данной сети будут подаваться информация о книге (Автор, Название, Часть каталога(жанр)) и пользовательском списке книг (то же самое что у книг + читал ли, понравилось ли, рейтинг) в виде матрицы.

Основной проблемой, которую я выявила на этапе продумывания, является то, что книг очень большое количество и каждый раз переобучать модель на все объекты будет затратно и выбрала два решения:
•	Как-то модифицировать модель
•	Использовать для обучения список наиболее рейтинговых или популярных книг

В данной модели, на первый взгляд, нет особой нужды в кластеризации входных данных, но для разделения книг на группы можно кластеризировать их по цене и рейтингу. В файле main.py представлена кластеризация книг с помощью k-means по цене и рейтингу.

Проблемы с которыми столкнулась:

• Большое многообразие различных подходов к реализации рекомендательных систем, среди которых нет наиболее общепризнаных алгоритмов - это достаточно сильно путает, но при дальнейшем погружении ситуация с ними проясняется

• У книг как будто не хватает различных атрибутов или атрибутов по которым можно их кластеризовать - изначально была задумка провести сегментацию еще и по автору или жанру, но при скролле датасета я выяснила, что у большинства имен автора имена уникальны так как содержат в себе название книги. Думаю скорее всего можно было бы как то вынести также жанры книг в кластеризирующие атрибуты, но так как многие из них уникальны, то не придумала как :(
