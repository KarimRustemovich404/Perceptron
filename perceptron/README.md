# Многослойный персептрон с нуля

Классификация опухолей молочной железы (M — злокачественная, B — доброкачественная) по 30 признакам ядер клеток.
Нейросеть реализована только на numpy: прямой проход, обратное распространение ошибки, мини-батч градиентный спуск, softmax на выходе.

## Установка
```
pip install -r requirements.txt
```

## Запуск
```
python eda.py --dataset data.csv                      # анализ данных, графики в eda/
python split.py --dataset data.csv --valid_ratio 0.2 --seed 42
python train.py --layer 24 24 24 --epochs 84 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.0314
python train.py --config network.json                 # то же, но через конфиг-файл
python predict.py --model saved_model.npz --dataset data/data_validation.csv
python gradient_check.py                              # проверка backprop численным градиентом
```

Параметры `train.py`: `--layer` (размеры скрытых слоёв, по умолчанию `24 24`), `--activation` (sigmoid, relu, tanh),
`--weights_initializer` (heUniform, heNormal, xavierUniform, xavierNormal, random), `--epochs`, `--batch_size`,
`--learning_rate`, `--loss`, `--patience` (ранняя остановка, 0 — выкл.), `--seed`, `--no_show`.

## Структура
```
mlp/
  activations.py    sigmoid, relu, tanh, softmax + их производные
  initializers.py   heUniform, heNormal, xavier...
  layers.py         DenseLayer: forward / backward / update
  losses.py         бинарная кросс-энтропия, MSE
  model.py          Network: fit, predict, save/load; createNetwork, fit в стиле задания
  preprocessing.py  загрузка CSV, стандартизация, one-hot
split.py            разделение на train/valid (стратифицированно, фиксированный seed)
train.py            обучение, вывод метрик по эпохам, сохранение модели, кривые обучения
predict.py          загрузка модели, прогноз, оценка бинарной кросс-энтропией
eda.py              статистический и визуальный анализ данных
gradient_check.py   численная проверка градиентов
network.json        пример конфигурации сети
```
