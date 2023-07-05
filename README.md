# supernet-mnist

Проводится эксперимент по обучению суперсети определённой архитектуры на датасете MNIST. На каждой итерации с помощью заданного типа сэмплирования выбирается подсеть в пространстве поиска, обучение которой и происходит. Рассматривается два типа сэмплирования: равномерное сэмплирование и сэмплирование на основе стохастического бандитского алгоритма Epsilon-Greedy. Также проводится независимое обучение подсетей. Оценка качества подсетей на тестовом датасете показывает, что имеет место высокая корреляция целевой метрики для подсетей, обученных в рамках суперсети и независимо. Это позволяет определять лучшие архитектуры при использовании меньшего количества ресурсов. Приводятся проблемы данного подхода и способы их решения.Подробности эксперимента содержатся в тетрадке.

## Архитектура SuperNet в эксперименте

SuperNet (one-shot networks, концепция weight sharing) включает в себя все сети из обозначенного пространства поиска, то есть по своей структуре эта большая сеть, в которой любая подсеть является сетью из зафиксированного пространства поиска.

Ниже показана архитектура SuperNet, которая покрывает пространство, состоящее из 9 архитектур (на картинке справа приведены примеры таких архитектур при изменении первого блока).

Свертки, находящиеся в изменяемом блоке (выделены желтым), не меняют высоту и ширину входного тензора, а также число каналов.  

![SuperNet architecture](pics/supernet.png "SuperNet architecture")

На схеме ниже показана архитектура изменяемого блока суперсети, а также ее конкретные реализации.

![NAS Block architecture](pics/NAS_block.png "NAS block architecture")

## Гипотезы

### Гипотеза №1

---

Поскольку все присутствуюшие в NAS-блоках подблоки разделяются между подсетями, то интуитивно первые из них могут служить хорошо предобученными для последующих подблоков, что может способствовать увеличению целевой метрики. То есть чем больше подблоков в NAS-блоке, тем лучше.

---

### Гипотеза №2

---

Поскольку `stride > 1` обычно применяется для повышения вычислительной производительности и способствует определённой потери информации, то чем больше подблоков в NAS-блоке, следующим за общим блоком (фиолетового цвета) с `stride = 2`, тем больше возможностей для восстановления важных признаков, утерянных из-за `stride`, что должно приводить к большей точности. То есть количество подблоков во втором NAS-блоке может быть ключевым фактором увеличения целевой метрики в пространстве поиска.

---

### Гипотеза №3

---

Количество подблоков в первом NAS-блоке также должно увеличивать целевую метрику, поскольку большее (но всё же ограниченное) число параметров позволяет выделить лучшие признаки перед фиолетовым блоком.

---

### Гипотеза №4

---

Лучшую архитектуру подсети можно попробовать выявить за меньшее количество эпох с помощью сэмплирования на основе алгоритма Epsilon-Greedy:
- для набора статистики в рамках первых `n` (100) раундом используется равномерное сэмплирование;
- в дальнейшем с вероятностью `ε` (0.2) используется равномерное сэмплирование, а в противном случае выбирается подсеть-лидер по целевой метрике.

---

Хотя Epsilon-Greedy концентрируется на лидере подсетей по Top-1 Acc, он также даёт возможность другим подсетям вырваться вперёд, что важно, поскольку все подсети разделяют часть весов друг с другом.

## Конфигурация пайплайна

## Значения метрик

Тип архитектуры подсети задаётся парой, компоненты которой определяют
количество подблоков первого и второго NAS-блока соответственно.

| Подсеть | Top-1 Acc на SuperNet | Top-1 Acc независимо |
|---------|-----------------------|----------------------|
| (1, 1)  |  0.853600             | 0.950733             |
| (1, 2)  |  0.941300             | 0.977233             |
| (1, 3)  |  0.951867             | 0.988917             |
| (2, 1)  |  0.935967             | 0.963383             |
| (2, 2)  |  0.962083             | 0.983217             |
| (2, 3)  |  0.968833             | 0.989100             |
| (3, 1)  |  0.947317             | 0.965817             |
| (3, 2)  |  0.969000             | 0.983167             |
| (3, 3)  |  0.971383             | 0.990533             |

Значения метрик показывают, что лучшие показатели соответствуют подсетям, имеющим 3 подблока во втором NAS-блоке.

### Выводы по эксперименту

- Наблюдается очень высокая корреляция между значениями метрик при обучении в рамках суперсети и независимо. Таким образом, использование суперсети эффективно с точки зрения использования ресурсов для определения лучшей архитектуры: потребовалось меньше времени для обучения суперсети, чем всех подсетей независимо.

- Вторая и третья гипотеза полностью подтверждены, причём количество подблоков во втором NAS-блоке является ключевым фактором роста целевой метрики.

### Потенциальные проблемы

1. Оценка подсетей, обученных в рамках суперсети, показала, что можно сделать ложный вывод о превосходстве одной архитектуры над другой, как это показал пример с архитектурами (1, 3) и (2, 2). Это может быть проблемой при выборе компромиссной архитектуры, когда требуется определить архитектуру приемлемого качества с меньшим или одинаковым количеством весов.

2. Равномерное сэмплирование не учитывает точность подсети при обучении, что может просто приводить к трате ресурсов на обучение заведомо плохих подсетей, что особенно критично, когда пространство поиска огромно.

### Подходы к решению проблем

Вторую проблему можно решать с помощью сужения пространства поиска при наборе статистики о подсетях в рамках обучения суперсети. Это может быть что-то вроде pruning, как для древесных алгоритмов.

Также можно уделять меньшее количество ресурсов на обучение подсетей, которые при наборе статистики показывают себя недостаточно хорошо. Это можно сделать с помощью сэмплирования на основе алгоритмов, учитывающих историю точности подсетей, например, стохастических бандитов или алгоритмов предсказания с помощью экспертов.

Первая проблема подталкивает к идее определения лидера среди подсетей и предоставления ему больших ресурсов при обучении суперсети.

Это делается во втором разделе тетрадки, где суперсеть обучается при алгоритме сэмплирования Epsilon-Greedy.

В процессе обучения суперсети Epsilon-Greedy уверенно выбирает лидера, а также другие подсети, у которых во втором NAS-блоке три подблока. Кроме того, по динамике точности видно, что сети хорошо кластеризуются (в отличие от ситуации равномерного сэмплирования), как по количеству блоков в первом, так и втором NAS-блоке.

Ещё про решение первой проблемы: конкурирующие подсети можно попробовать выделить из суперсети и дообучить несколько эпох, чтобы определить лучшую. Пока этот эксперимент не проводится в тетрадке.