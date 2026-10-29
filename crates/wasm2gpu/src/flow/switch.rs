//! `br_table` to a naga `switch`.
//!
//! Each arm sets the same leave/again flags as `br`. A `break` inside the
//! switch would leave the switch, not the wasm block, so the arm does not
//! break the enclosing loop itself. [`Emit::flush_top`] does that after the
//! switch. Restarting a loop is a `continue`, which already targets the loop
//! from inside the switch.

use crate::cx::sp;
use crate::env::Emit;
use crate::env::Region;
use crate::error::TranslateError;

impl Emit<'_, '_> {
    pub(crate) fn br_table(&mut self, targets: &[u32], default: u32) -> Result<(), TranslateError> {
        let selector = self.pop_uint()?;
        let mut cases = Vec::new();
        let mut index = 0u32;
        while (index as usize) < targets.len() {
            let arm = self.case(naga::SwitchValue::U32(index), targets[index as usize])?;
            cases.push(arm);
            index += 1;
        }
        cases.push(self.case(naga::SwitchValue::Default, default)?);
        self.block
            .push(naga::Statement::Switch { selector, cases }, sp());
        self.flush_top()
    }

    fn case(
        &mut self,
        value: naga::SwitchValue,
        depth: u32,
    ) -> Result<naga::SwitchCase, TranslateError> {
        let body = self.sub(|em| em.switch_jump(depth))?;
        Ok(naga::SwitchCase {
            value,
            body,
            fall_through: false,
        })
    }

    fn switch_jump(&mut self, depth: u32) -> Result<(), TranslateError> {
        let n = self.labels.len();
        if depth as usize == n {
            return Err(TranslateError::FunctionBranch);
        }
        let index = n
            .checked_sub(depth as usize + 1)
            .ok_or(TranslateError::Unsupported)?;
        let target = self
            .labels
            .get(index)
            .copied()
            .ok_or(TranslateError::Unsupported)?;
        if depth == 0 && target.kind == Region::Loop && target.naga_loop {
            self.block.push(naga::Statement::Continue, sp());
            return Ok(());
        }
        if depth == 0 && !target.naga_loop {
            return Err(TranslateError::Unsupported);
        }
        self.flag_outers(index)
    }
}
