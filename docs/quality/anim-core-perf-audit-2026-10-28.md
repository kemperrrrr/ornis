# Перф-аудит ядра анимации (фаза E) — 2026-10-28

Статическая проверка горячего пути `crates/animation` на head `33d6fac`
(после р-31: дерево, машины, switch, eff-веса, события, IK, root motion,
ретаргет). Замеры кадров — за ревьюверами (бенчи `#[ignore]`); здесь —
инвентарь стоимостей и точек проверки, по которым читаются профили.
Бэклог `docs/animation-design.md` §5 п.5 закрыт этим аудитом (пункт
отмечен сделанным там же).

## Метод

- Полный проход по `crates/animation/src/*.rs` вне `#[cfg(test)]` на
  аллокации (`vec!`, `Vec::new`, `.collect`, `.to_vec`, `.clone`,
  `HashMap`/`HashSet`/`String`).
- Разбор каждого прохода кадра: маршрутизация, семпл, бленд, шаг курсоров,
  события, фейды, eff-walk, IK, root motion, публикация.
- Сверка покрытия буферов `AnimScratch` против фактических точек записи.

## Итог

**Разовый кадр при устоявшейся сцене не аллоцирует.** Каждый проход пишет
в `AnimScratch` (`crates/animation/src/lib.rs:599`) или в собственные `Vec`
компонентов; снапшот frozen — битовая маска `FrozenBits` (`lib.rs:561`),
каждый skip-тест O(1) (`FrozenBits::rebuild` переиспользует буфер).

| Проход | Точки | Буферы | Аллокации на кадр |
| --- | --- | --- | --- |
| Объектный семпл | `sample_and_publish` (`lib.rs:751`), `track_of` (`:863`), `side_pose` (`:869`) | `obj_work`/`obj_locals`, `pose_a`/`pose_b` | нет |
| Скелетный семпл | `collect_pose_work` (`:2809`), `sample_entity` (`:3007`), `blend_skel_pose` (`:3035`), `overlay_layers` (`:3083`), `finish_pose` (`:2953`) | `joints`, `contributors`, `pose_a`/`pose_b`, `skel_locals`, `matrices`, `nodes`, `node_painted` | нет |
| Дерево | `sample_tree_entity` (`:3125`), `eval_tree` (`tree_eval.rs`), `step_machines` (`:2502`), `step_switches` (`:2564`) | `tree_poses` (`fit_tree_poses` `:2914`, рост-only), `tree_weights`, курсорная арифметика | нет |
| Eff-веса | `leaf_effective_weights` (`tree_eff.rs:47`) | `WeightsScratch` | нет |
| События | `emit_step` (`events.rs:176`), `fire_step` (`:143`) | `queue.fired` (компонент) | только на срабатываниях (см. ниже) |
| Фейды | `tick_fades` (`lib.rs:2682`), `share_of` (`:2673`) | `finished`, `FrozenBits` | нет |
| IK | `apply_ik_stack` (`ik.rs:283`) | `ik_fk` (`FkScratch`) | нет |
| Root motion | хвост `finish_pose`, `apply_ragdoll` (`:2928`) | `layer_motion`, `motion_hold` | нет |
| Скин-публикация | `finish_pose` → `compose_model_matrices` (`:1906`) | `matrices`, `pose.matrices` (компонент) | нет |

## Принятые аллокации вне кадра

Частота события (не кадр):

- `fire_step` (`events.rs:154`) — `hits: Vec<(f32, usize)>` берёт память
  только когда метки реально пересекаются (на кадрах без пересечений
  `Vec::new()` не аллоцирует); имя события — `String`-клон на каждое
  срабатывание (`AnimEventOut.event`, публичный тип).
- `sync_transition_leaves` (`tree_sync.rs:165`) — `subtree_leaves`
  (`:102`: `vec![false; nodes]` + список листьев) и `leaving_clocks`
  (`:136`) аллоцируются на старте перехода машины/switch, не на кадр.

Частота команда (хост пишет состояние):

- `ControlRig::into_stack` (`ik.rs:169`) — на запись IK-целей.
- `Animator` — `targets.clone()` на `play`/`play_crossfade`.
- `RetargetMap::bake` (`retarget.rs:50`) и `SkinningResources::build`
  (`skinning.rs:193`) — на установку карты/палитры.

## Границы сложности (когда трогать)

- `track_of` (`lib.rs:863`) — линейный поиск трека: O(tracks) на сущность
  за кадр (×2 с фейдом). Для клипов «трек на сущность» это O(E) по клипу и
  незаметно; для толповых клипов с десятками треков станет O(E×T) — тогда
  индекс (entity → слот трека) на загрузке клипа.
- `subtree_leaves` (`tree_sync.rs:102`) — O(nodes²) проверок
  `node_references` (`:77`) на переход; узлы машин — единицы-десятки,
  переходы — частота геймплея.
- `node_poses` (`lib.rs:3170`) — дедуп `out[start..]` линеен по записям
  кадра и ограничен `MAX_JOINTS`.

## Точки замера (ревьюверам)

- Кадровый бенч: `crates/animation/tests/hot_path_bench.rs` (сценарии
  `empty`/`small`/`skel_heavy`/`tree_heavy`/`machine_heavy`/`object_heavy`/
  `full_mix`); A/B против base — `crates/animation/tests/hot_path_ab.sh`.
- Проверка «аллокаций нет» в профиле: любой кадр после прогрева — доли
  `alloc` в `sample_and_publish`/`collect_pose_work` должны быть на уровне
  шума; всплеск = смотреть таблицу «Принятые аллокации» (событие или
  команда, но не кадр).
