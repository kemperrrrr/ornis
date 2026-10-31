//! Tukey hinges over one timed series.
//!
//! Even-length medians are the mean of the two central samples. The lower
//! hinge is the median of the lower half and the upper hinge is the median
//! of the upper half; on an odd length the central sample is in neither half.

/// One series after sorting: quartiles, extremes, and the sample count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Summary {
    /// Samples that entered the summary.
    pub n: usize,
    /// Smallest sample.
    pub min: f64,
    /// Lower hinge.
    pub q1: f64,
    /// Median.
    pub median: f64,
    /// Upper hinge.
    pub q3: f64,
    /// `q3 - q1`.
    pub iqr: f64,
    /// Largest sample.
    pub max: f64,
}

/// Sort `samples` and return the Tukey summary. An empty series is all zeros.
pub(crate) fn summarize(samples: &[f64]) -> Summary {
    assemble(&sorted(samples))
}

fn sorted(samples: &[f64]) -> Vec<f64> {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered
}

fn assemble(ordered: &[f64]) -> Summary {
    let median = median_of(ordered);
    let q1 = median_of(lower_half(ordered));
    let q3 = median_of(upper_half(ordered));
    Summary {
        n: ordered.len(),
        min: edge(ordered, true),
        q1,
        median,
        q3,
        iqr: q3 - q1,
        max: edge(ordered, false),
    }
}

fn median_of(ordered: &[f64]) -> f64 {
    let n = ordered.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        ordered[n / 2]
    } else {
        (ordered[n / 2 - 1] + ordered[n / 2]) / 2.0
    }
}

fn lower_half(ordered: &[f64]) -> &[f64] {
    &ordered[..ordered.len() / 2]
}

fn upper_half(ordered: &[f64]) -> &[f64] {
    let mid = ordered.len() / 2;
    &ordered[ordered.len() - mid..]
}

fn edge(ordered: &[f64], low: bool) -> f64 {
    if ordered.is_empty() {
        0.0
    } else if low {
        ordered[0]
    } else {
        ordered[ordered.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::summarize;

    #[test]
    fn even_length_uses_the_mean_of_the_two_central_samples() {
        let samples: Vec<f64> = (1..=8).map(|value| value as f64).collect();
        let summary = summarize(&samples);
        assert_eq!(summary.n, 8);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.q1, 2.5);
        assert_eq!(summary.median, 4.5);
        assert_eq!(summary.q3, 6.5);
        assert_eq!(summary.iqr, 4.0);
        assert_eq!(summary.max, 8.0);
    }

    #[test]
    fn odd_length_keeps_the_median_out_of_both_halves() {
        let samples: Vec<f64> = (1..=5).map(|value| value as f64).collect();
        let summary = summarize(&samples);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.q1, 1.5);
        assert_eq!(summary.q3, 4.5);
        assert_eq!(summary.iqr, 3.0);
    }

    #[test]
    fn unsorted_input_is_ordered_before_the_hinges() {
        let summary = summarize(&[4.0, 1.0, 3.0, 2.0]);
        assert_eq!(summary.median, 2.5);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 4.0);
    }

    #[test]
    fn empty_series_is_zero() {
        let summary = summarize(&[]);
        assert_eq!(summary.n, 0);
        assert_eq!(summary.median, 0.0);
        assert_eq!(summary.iqr, 0.0);
    }
}
