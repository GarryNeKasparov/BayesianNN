# BayesianNN
Курсовая работа по численным методам

# Структура проекта
Методы Монте-Карло:
* [mc_integration.py](https://github.com/GarryNeKasparov/BayesianNN/blob/main/bayesiannn/monte_carlo/mc_integration.py) - интегрирование методом Монте-Карло; 
* [regression_example.py](https://github.com/GarryNeKasparov/BayesianNN/blob/main/bayesiannn/monte_carlo/regression_example.py) - реализация метода Метрополиса-Гастингса;
Вариационный вывод:
* [course.ipynb](https://github.com/GarryNeKasparov/BayesianNN/blob/main/bayesiannn/app/course.ipynb) - ноутбук с построением и обучением модели;
* [main.py](https://github.com/GarryNeKasparov/BayesianNN/blob/main/bayesiannn/app/main.py) - интерфейс для работы с приложением.

# Запуск приложения
Установить необходимые зависимости, используя:
```
poetry install
```

Находясь в директории проекта, выполнить:
```
cd app
python -m fastapi run
