#![no_main]

//! Фаззинг десериализации сцены из RON — внешний ввод (файлы сцен,
//! приходящие от редактора и с диска). Парсер не должен паниковать
//! ни на каком входе: любой мусор → Ok или Err, но не panic/abort.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = ornis_render::scene::Scene::from_ron(s);
    }
});
