use syn::{visit, visit::Visit};

#[derive(Default)]
pub struct StaticProfile {
    pub compute_ops: usize,
    pub branch_ops: usize,
    pub total_ops: usize,
}

#[allow(dead_code)]
impl StaticProfile {
    pub fn branch_ratio(&self) -> f64 {
        if self.total_ops == 0 {
            return 0.0;
        }
        self.branch_ops as f64 / self.total_ops as f64
    }

    pub fn prefers_gpu(&self) -> bool {
        self.branch_ratio() <= 0.05
    }
}

impl<'ast> Visit<'ast> for StaticProfile {
    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        use syn::BinOp::*;
        match node.op {
            Add(_) | Sub(_) | Mul(_) | Div(_) | Rem(_) => {
                self.compute_ops += 1;
                self.total_ops += 1;
            }
            _ => {}
        }
        visit::visit_expr_binary(self, node);
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        self.branch_ops += 1;
        self.total_ops += 1;
        visit::visit_expr_if(self, node);
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        self.branch_ops += 1;
        self.total_ops += 1;
        visit::visit_expr_match(self, node);
    }
}

#[allow(dead_code)]
pub fn analyze(func: &syn::ItemFn) -> StaticProfile {
    let mut profile = StaticProfile::default();
    profile.visit_item_fn(func);
    profile
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn pure_math_prefers_gpu() {
        let func: syn::ItemFn = parse_quote!(
            fn kernel(a: f32, b: f32) -> f32 {
                a * b + sin(a) * cos(b)
            }
        );
        let profile = analyze(&func);
        assert!(profile.prefers_gpu());
        assert!(profile.branch_ratio() < 0.05);
    }

    #[test]
    fn heavy_branching_prefers_cpu() {
        let func: syn::ItemFn = parse_quote!(
            fn kernel(a: f32) -> f32 {
                if a > 0.0 {
                    if a > 10.0 {
                        a * 2.0
                    } else {
                        a
                    }
                } else {
                    a * 3.0 + if a < -10.0 { 1.0 } else { 0.0 }
                }
            }
        );
        let profile = analyze(&func);
        assert!(!profile.prefers_gpu());
        assert!(profile.branch_ratio() > 0.05);
    }

    #[test]
    fn single_branch_still_prefers_gpu() {
        let func: syn::ItemFn = parse_quote!(
            fn kernel(a: f32) -> f32 {
                if a > 0.0 { a * 2.0 + b * 3.0 + c * 4.0 + d * 5.0 + e * 6.0 + f * 7.0 + g * 8.0 + h * 9.0 + i * 10.0 + j * 11.0 + k * 12.0 + l * 13.0 + m * 14.0 + n * 15.0 + o * 16.0 + p * 17.0 + q * 18.0 + r * 19.0 + s * 20.0 } else { a }
            }
        );
        let profile = analyze(&func);
        assert!(profile.prefers_gpu());
    }
}
