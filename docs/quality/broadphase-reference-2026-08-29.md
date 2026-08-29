# Broadphase: разбор Box3D и Jolt — 2026-08-29

## Зачем это зафиксировано

После первых замеров `UniformGrid` возникло справедливое подозрение, что
подбор `cell_size` оптимизирует не тот уровень проблемы. Поэтому были
просмотрены актуальные upstream-исходники Box3D и Jolt Physics. Цель этого
документа — зафиксировать архитектурные выводы, а не объявить новый backend
production-ready.

## Короткий вывод

Мы не изобретаем саму broadphase-задачу, но текущая реализация Ornis пока
проще зрелых решений в важном месте: `UniformGrid` пересобирает swept AABB,
cell memberships и candidate pairs для всех тел на каждом substep.

Box3D и Jolt делают основной упор не на поиске идеального размера ячейки, а
на следующих свойствах:

- persistent broadphase proxies;
- fat AABB и дешёвые incremental updates;
- отдельные структуры для static и moving bodies или broadphase layers;
- active/moved body list вместо полного pair pass;
- периодическая оптимизация дерева, а не полная перестройка на каждом запросе;
- ранняя фильтрация и переиспользование уже построенного broadphase state.

Следовательно, adaptive `cell_size` остаётся полезным tuning-инструментом, но
не является заменой persistent dynamic broadphase.

## Что найдено в Box3D

Официальное исходное дерево Box3D содержит отдельные
[`broad_phase.c`](https://github.com/erincatto/box3d/blob/main/src/broad_phase.c),
[`broad_phase.h`](https://github.com/erincatto/box3d/blob/main/src/broad_phase.h)
и
[`dynamic_tree.c`](https://github.com/erincatto/box3d/blob/main/src/dynamic_tree.c).

### Структура

`b3BroadPhase` держит несколько dynamic AABB trees — по одному для типов
static, kinematic и dynamic proxies. Proxy имеет стабильный индекс в дереве и
пользовательские category bits.

### Обновление и pair generation

- Движения буферизуются через bitsets `movedProxies` и упорядоченный
  `moveArray`.
- Static proxy обычно не добавляется в move buffer при обычном создании;
  dynamic proxy запрашивает только релевантные деревья.
- Pair query запускается для moved proxies, а не для всех возможных пар.
- Уже существующие контакты отслеживаются отдельным `pairSet`, поэтому
  broadphase сообщает новые потенциальные пары, а lifecycle контактов не
  смешивается с обходом дерева.
- Параллельные pair tasks используют deterministic порядок move buffer;
  перестройка dynamic/kinematic trees может выполняться рядом с последующей
  contact work.

### Dynamic tree

Внутри используется binary dynamic AABB tree с greedy SAH-подобным выбором
sibling и rotations. Proxy AABB может расширять путь родителей без немедленной
полной перестройки; отдельная операция rebuild делает дерево более tight и
может переиспользовать неизменившиеся поддеревья.

Это важное отличие от текущего Ornis grid: стоимость перемещения тела не
равна стоимости заново проиндексировать весь мир.

## Что найдено в Jolt

Jolt описывает broadphase в своей
[архитектуре collision detection](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md).
Реализация находится в
[`BroadPhaseQuadTree`](https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/Collision/BroadPhase/BroadPhaseQuadTree.h)
и внутреннем
[`QuadTree`](https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/Collision/BroadPhase/QuadTree.h).

### Broadphase layers

Jolt не складывает весь мир в одну универсальную структуру. Object layers
отображаются в несколько broadphase layers, обычно как минимум:

- `NON_MOVING` для static geometry;
- `MOVING` для обычных dynamic bodies.

При необходимости добавляются отдельные категории вроде debris или bullet
bodies. Фильтр broadphase layer применяется до более дорогого object-layer
filtering, поэтому структуру можно подстроить под реальные паттерны
столкновений.

### QuadTree и обновление

- Внутренний узел имеет до четырёх children и хранит bounds в SIMD-friendly
  SoA-раскладке; traversal проверяет четыре AABB за раз.
- Для каждого body сохраняется tracking его позиции в дереве.
- `NotifyBodiesAABBChanged` помечает изменившийся путь, а
  `UpdatePrepare`/`UpdateFinalize` обновляют дерево отдельно от обычных
  queries.
- Неизменившиеся части дерева переиспользуются; периодическая перестройка
  возвращает tight bounds.
- Деревья double-buffered относительно queries, чтобы update и чтение не
  требовали глобальной остановки мира.
- `FindCollidingPairs` получает список active bodies, запрашивает их AABB в
  relevant trees, делает финальную AABB-проверку и применяет object-layer
  filter.

Jolt сложнее и сильнее ориентирован на многопоточность, locking и SIMD, но
главный переносимый вывод тот же: persistent tree + active-body queries важнее
автоматического подбора единственного scalar cell size.

## Сопоставление с текущим Ornis

| Аспект | Ornis сейчас | Box3D | Jolt |
|---|---|---|---|
| Основная структура | `HashMap<CellKey, Vec<usize>>` | dynamic AABB tree по body type | 4-way QuadTree по broadphase layer |
| Lifetime proxy | фактически пересоздаётся при update | persistent proxy | persistent tracking/proxy |
| Обновление | swept AABB и все memberships заново | moved/enlarged proxy, периодический rebuild | changed paths, `UpdatePrepare`/`Finalize` |
| Pair search | occupants всех занятых cells | moved proxies против relevant trees | active bodies против relevant trees |
| Static geometry | тот же grid, static-static skip при pair test | отдельное static tree | отдельный non-moving layer/tree |
| Filtering | collision layer/mask до narrowphase | category/body/custom filters | broadphase layer + object layer filters |
| Deduplication | `HashSet` на каждом update | contact/pair set отдельно | collector/contact lifecycle отдельно |
| SIMD/parallel broadphase | пока отсутствуют | parallel pair/tree tasks | SIMD traversal и multicore-safe update |

У Ornis уже есть полезные свойства — deterministic sorted pairs, collision
layers/masks, static-static optimization, sleeping и solver warm cache. Но
`warm_impulses` — это cache solver-а, не persistent broadphase pair cache.

Также важно уточнить терминологию: текущий grid не имеет двух независимых
static/dynamic cell maps. Все тела вставляются в общий набор cells, а
static-static пары отбрасываются после генерации raw pair.

## Решение для Ornis

### Оставить сейчас

1. `Sweep-and-Prune` остаётся default baseline/fallback.
2. `UniformGrid` остаётся opt-in backend и полезным специализированным
   вариантом для tiled/static worlds.
3. Расширенный manual pass уже проверил `cell_size = 8.0/16.0`: для текущей
   tiled-floor сцены лучший измеренный вариант — `8.0`, но он остаётся только
   provisional tuning choice.
4. Production default не меняется по одной tiled-floor сцене и одному 10k
   прогону.

### Следующий архитектурный кандидат

Добавить экспериментальный `DynamicAabbTree` backend, перенеся идеи, а не
копируя исходный C/C++ код:

```text
persistent proxy per body
fat AABB
static tree + dynamic tree
moved/awake body list

for each active dynamic body:
    query dynamic tree
    query static tree
    final swept-AABB check
    collision-layer filtering
    canonicalize pairs
```

Для Ornis особенно важны обработка `swap_remove` body handles, deterministic
порядок результатов, wake-up sleeping bodies при контакте с moving body и
корректное удаление/remap proxy.

### Что делать с adaptive Grid

Adaptive policy не удаляется из roadmap, но откладывается до persistent
backend:

- не менять размер каждый кадр;
- переобучать только при существенном изменении body distribution;
- оценивать несколько кандидатов по wall-clock или calibrated cost model;
- учитывать memberships, occupied cells, pair tests, large-body fallback и
  стоимость rebuild;
- использовать hysteresis, чтобы не переключаться между соседними размерами;
- рассматривать HGrid/hierarchical spatial hash для сильно неоднородных
  размеров, а не только один scalar `cell_size`.

Практически adaptive grid может остаться специализированным static/terrain
ускорителем, а dynamic heterogeneous world будет обслуживаться деревом.

## План проверки

1. ✅ Получить фактические результаты для grid `8/16` на 10k: `8.0` оказался
   лучшим измеренным вариантом, `16.0` уже хуже.
2. ✅ 100k targeted probes выполнены для Grid 8.0 (`33245718111`, около
   `8.02 s/step`) и SAP (`33251548032`, около `79.49 s/step`); Grid примерно
   в 9.9x быстрее SAP в steady state на tiled-floor workload.
3. Добавить timing breakdown broadphase / narrowphase / solver.
4. Реализовать маленький корректный `DynamicAabbTree` модуль с brute-force
   oracle-тестами.
5. Сравнить tree, grid и SAP на tiled floor, giant floor, sparse world,
   dense islands и heterogeneous body sizes.
6. Только после этого принимать решение о default или adaptive routing.

## Границы и лицензии

Это архитектурное заимствование идей, не копирование реализации. Box3D и
Jolt распространяются с permissive MIT-лицензией; при прямом reuse кода нужно
сохранить исходные copyright/license notices. Для Ornis предпочтительно
написать native Rust implementation с теми же проверенными принципами и
сохранить собственную deterministic/testable API.

## Источники

- [Box3D repository](https://github.com/erincatto/box3d)
- [Box3D broad phase source](https://github.com/erincatto/box3d/blob/main/src/broad_phase.c)
- [Box3D dynamic tree source](https://github.com/erincatto/box3d/blob/main/src/dynamic_tree.c)
- [Jolt collision architecture](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md)
- [Jolt BroadPhaseQuadTree header](https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/Collision/BroadPhase/BroadPhaseQuadTree.h)
- [Jolt QuadTree header](https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/Collision/BroadPhase/QuadTree.h)
- [Jolt BroadPhase API documentation](https://jrouwe.github.io/JoltPhysics/class_broad_phase.html)

Источники просмотрены 2026-08-29; ссылки указывают на upstream `main`/`master`
и могут изменяться вместе с проектами.
