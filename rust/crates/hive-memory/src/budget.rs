//! Adaptive context budget (focal layer).

pub struct AdaptiveBudget {
    pub ultra_small_budget_tokens: u32,
}

impl Default for AdaptiveBudget {
    fn default() -> Self {
        Self {
            ultra_small_budget_tokens: 1000,
        }
    }
}

impl AdaptiveBudget {
    pub fn compute(&self, route: &str, high_relevance_count: usize, max_context: u32) -> u32 {
        let (mut lo, hi) = match route {
            "ultra_small" => (1000, 3000),
            "medium" => (3000, 5000),
            "escalation" => (4000, 6000),
            _ => (2000, 4000),
        };
        if route == "ultra_small" {
            lo = self.ultra_small_budget_tokens;
        }
        let fill = (high_relevance_count as f64 / 10.0).min(1.0);
        let budget = (lo as f64 + (hi - lo) as f64 * fill).floor() as u32;
        budget.min(max_context.saturating_sub(2048))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_fills_with_high_relevance_and_reserves_headroom() {
        let budget = AdaptiveBudget::default();
        assert_eq!(budget.compute("ultra_small", 0, 8192), 1000);
        assert_eq!(budget.compute("ultra_small", 10, 8192), 3000);
        assert_eq!(budget.compute("ultra_small", 10, 4096), 2048);
        assert_eq!(budget.compute("escalation", 0, 8192), 4000);
    }
}
