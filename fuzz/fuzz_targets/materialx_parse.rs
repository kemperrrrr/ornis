#![no_main]

//! Фаззинг MaterialX XML-парсера — внешний ввод (.mtlx ассеты).
//! Инвариант: любой вход → Ok/Err, без паник и бесконечных циклов.

use libfuzzer_sys::fuzz_target;
use ornis_materialx::MaterialXParser;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let parser = MaterialXParser::new();
        let _ = parser.parse(s);
    }
});
