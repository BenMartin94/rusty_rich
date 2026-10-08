//! Functional test: the forward solver against the exact series solution for a
//! homogeneous dielectric circular cylinder illuminated by a line source.
//!
//! This checks the whole `solver::solve` path - kernel, GMRES, point-source
//! incident field and receiver integration - against an independent analytic
//! answer, rather than against another numerical solver.

use complex_bessel::{besselj, hankel2};
use num_complex::Complex;
use rusty_rich::solver::{self, Domain, IncidentWave, Medium, SolveRequest, PI};

type C = Complex<f64>;

const FREQUENCY_HZ: f64 = 5e9;
const CYLINDER_EPS: f64 = 2.3;
const CYLINDER_RADIUS: f64 = 0.03;
const EXTENT: f64 = 0.2;
/// Antennas sit on a ring outside the imaging domain, as in the measurement setup.
const RING_RADIUS: f64 = 0.144;
const NUM_RECEIVERS: usize = 24;
/// Source angles (degrees) on the ring; deliberately not aligned with the grid axes.
const SOURCE_ANGLES_DEG: [f64; 2] = [0.0, 37.0];

fn jn(n: i32, z: C) -> C {
    besselj(n as f64, z).unwrap()
}

fn hn(n: i32, z: C) -> C {
    hankel2(n as f64, z).unwrap()
}

/// Z_n'(z) = (Z_{n-1}(z) - Z_{n+1}(z)) / 2, valid for J and H.
fn jn_prime(n: i32, z: C) -> C {
    (jn(n - 1, z) - jn(n + 1, z)) * 0.5
}

fn hn_prime(n: i32, z: C) -> C {
    (hn(n - 1, z) - hn(n + 1, z)) * 0.5
}

/// Exact scattered field of a cylinder of radius `radius` and relative
/// permittivity `eps` centred at the origin, for the unit line source
/// `-j/4 H0^(2)(k0 |r - r_s|)` used by the solver (e^{jwt} convention).
/// `src` and `obs` are polar (rho, phi) and must both lie outside the cylinder.
fn analytic_scattered(k0: f64, eps: f64, radius: f64, src: (f64, f64), obs: (f64, f64)) -> C {
    let k0c = C::new(k0, 0.0);
    let k1c = C::new(k0 * eps.sqrt(), 0.0);
    let (rho_s, phi_s) = src;
    let (rho, phi) = obs;
    let n_max = (k1c.re * radius).ceil() as i32 + 25;

    // Expand the incident field as sum_n J_n(k0 rho) H_n(k0 rho_s) e^{jn(phi - phi_s)},
    // write the scattered field as sum_n a_n H_n(k0 rho) e^{jn(phi - phi_s)}, and
    // enforce continuity of u and du/drho at rho = radius for each order n.
    let coeff = |n: i32| -> C {
        let (x0, x1) = (k0c * radius, k1c * radius);
        let num = k0c * jn_prime(n, x0) * jn(n, x1) - k1c * jn(n, x0) * jn_prime(n, x1);
        let den = k0c * hn_prime(n, x0) * jn(n, x1) - k1c * hn(n, x0) * jn_prime(n, x1);
        -hn(n, k0c * rho_s) * num / den
    };

    // a_{-n} H_{-n} = a_n H_n, so the series folds into a cosine sum.
    let mut sum = coeff(0) * hn(0, k0c * rho);
    for n in 1..=n_max {
        sum += coeff(n) * hn(n, k0c * rho) * 2.0 * (n as f64 * (phi - phi_s)).cos();
    }
    C::new(0.0, -0.25) * sum
}

/// Rasterises the cylinder onto an n x n grid, weighting each cell by the
/// fraction of its area inside the circle (sub-sampled) to avoid staircase error.
fn cylinder_permittivity(domain: &Domain, center: (f64, f64)) -> Vec<C> {
    const SUB: usize = 8;
    let (dx, dy) = (domain.dx(), domain.dy());
    let mut eps = Vec::with_capacity(domain.cells());
    for i in 0..domain.nx {
        for j in 0..domain.ny {
            let mut inside = 0;
            for si in 0..SUB {
                for sj in 0..SUB {
                    let x = dx * (i as f64 - 0.5 + (si as f64 + 0.5) / SUB as f64);
                    let y = dy * (j as f64 - 0.5 + (sj as f64 + 0.5) / SUB as f64);
                    if (x - center.0).powi(2) + (y - center.1).powi(2) <= CYLINDER_RADIUS.powi(2) {
                        inside += 1;
                    }
                }
            }
            let fraction = inside as f64 / (SUB * SUB) as f64;
            eps.push(C::new(1.0 + (CYLINDER_EPS - 1.0) * fraction, 0.0));
        }
    }
    eps
}

/// Receivers weaker than this fraction of the peak scattered field sit near
/// nulls of the pattern, where |numeric/exact| and arg(numeric/exact) are
/// dominated by tiny absolute errors; magnitude/phase are only judged above it.
/// Every receiver is still covered by `max_pointwise`.
const STRONG_RECEIVER_FRACTION: f64 = 0.1;

/// Solver vs analytic series over every source/receiver pair.
#[derive(Debug)]
struct Errors {
    /// Relative L2 error of the complex field, ||numeric - exact|| / ||exact||.
    complex: f64,
    /// Worst complex (re and im) error at any single receiver, relative to the
    /// peak field: max |numeric - exact| / max |exact|.
    max_pointwise: f64,
    /// Worst relative magnitude error, max | |numeric| / |exact| - 1 |, over strong receivers.
    max_magnitude: f64,
    /// Worst phase error, max |arg(numeric / exact)|, degrees, over strong receivers.
    max_phase_deg: f64,
    /// Signed mean phase error over strong receivers, degrees (systematic bias).
    mean_phase_deg: f64,
}

impl Errors {
    fn from_pairs(pairs: &[(C, C)]) -> Self {
        let peak = pairs.iter().map(|(_, exact)| exact.norm()).fold(0.0, f64::max);
        let err_sq: f64 = pairs.iter().map(|(numeric, exact)| (numeric - exact).norm_sqr()).sum();
        let ref_sq: f64 = pairs.iter().map(|(_, exact)| exact.norm_sqr()).sum();
        let max_pointwise = pairs.iter().map(|(numeric, exact)| (numeric - exact).norm()).fold(0.0, f64::max) / peak;

        let strong: Vec<C> = pairs
            .iter()
            .filter(|(_, exact)| exact.norm() >= STRONG_RECEIVER_FRACTION * peak)
            .map(|(numeric, exact)| numeric / exact)
            .collect();
        let max_magnitude = strong.iter().map(|r| (r.norm() - 1.0).abs()).fold(0.0, f64::max);
        let max_phase_deg = strong.iter().map(|r| r.arg().to_degrees().abs()).fold(0.0, f64::max);
        let mean_phase_deg = strong.iter().map(|r| r.arg().to_degrees()).sum::<f64>() / strong.len() as f64;

        Errors { complex: (err_sq / ref_sq).sqrt(), max_pointwise, max_magnitude, max_phase_deg, mean_phase_deg }
    }
}

/// Compares the solver against the analytic series on an n x n grid.
fn errors(n: usize) -> Errors {
    let domain = Domain { nx: n, ny: n, x_meters: EXTENT, y_meters: EXTENT };
    // Cell (i, j) sits at (dx*i, dy*j), so the grid's geometric centre is here.
    let center = (domain.dx() * (n - 1) as f64 / 2.0, domain.dy() * (n - 1) as f64 / 2.0);
    let medium = Medium { frequency_hz: FREQUENCY_HZ, background_permittivity: C::new(1.0, 0.0) };
    let k0 = medium.k_b().re;
    let permittivity = cylinder_permittivity(&domain, center);

    let on_ring = |angle: f64| (center.0 + RING_RADIUS * angle.cos(), center.1 + RING_RADIUS * angle.sin());
    let receiver_angles: Vec<f64> = (0..NUM_RECEIVERS).map(|r| 2.0 * PI * r as f64 / NUM_RECEIVERS as f64).collect();
    let receivers: Vec<(f64, f64)> = receiver_angles.iter().map(|&a| on_ring(a)).collect();

    let mut pairs = Vec::new();
    for &src_deg in &SOURCE_ANGLES_DEG {
        let src_angle = src_deg.to_radians();
        let (sx, sy) = on_ring(src_angle);
        let result = solver::solve(&SolveRequest {
            domain,
            medium,
            permittivity: permittivity.clone(),
            incident: IncidentWave::PointSource { x: sx, y: sy },
            receivers: receivers.clone(),
            restart: 30,
            max_iter: 1000,
            tol: 1e-8,
        })
        .expect("solve failed");

        for (&numeric, &rx_angle) in result.receiver_scattered.iter().zip(&receiver_angles) {
            let exact = analytic_scattered(k0, CYLINDER_EPS, CYLINDER_RADIUS, (RING_RADIUS, src_angle), (RING_RADIUS, rx_angle));
            pairs.push((numeric, exact));
        }
    }
    Errors::from_pairs(&pairs)
}

/// Measured at 100x100: complex 0.73%, worst receiver 0.60% of peak, worst
/// magnitude 1.4%, worst phase 1.2 deg, mean phase bias +0.51 deg.
#[test]
fn matches_analytic_cylinder() {
    let e = errors(100);
    println!("100x100 vs analytic: {e:?}");
    assert!(e.complex < 0.01, "complex relative error {:.5} exceeds 1%", e.complex);
    assert!(e.max_pointwise < 0.01, "complex error at some receiver is {:.5} of peak, exceeds 1%", e.max_pointwise);
    assert!(e.max_magnitude < 0.02, "magnitude error {:.5} exceeds 2% at a strong receiver", e.max_magnitude);
    assert!(e.max_phase_deg < 2.0, "phase error {:.3} deg exceeds 2 deg at a strong receiver", e.max_phase_deg);
    assert!(e.mean_phase_deg.abs() < 1.0, "mean phase bias {:.3} deg exceeds 1 deg", e.mean_phase_deg);
}

/// The pulse-basis Richmond method is second order in cell size, so halving
/// dx should cut every error measure by ~4x (measured 50 -> 100 -> 200:
/// complex 3.1% -> 0.73% -> 0.19%, phase bias 2.2 -> 0.51 -> 0.13 deg).
/// Require at least 3x so a lost order - or a bias that does not shrink,
/// such as a position offset - fails the test.
#[test]
fn converges_at_second_order() {
    let coarse = errors(50);
    let fine = errors(100);
    println!("50x50: {coarse:?}\n100x100: {fine:?}");
    let ratios = [
        ("complex", coarse.complex / fine.complex),
        ("max_pointwise", coarse.max_pointwise / fine.max_pointwise),
        ("max_magnitude", coarse.max_magnitude / fine.max_magnitude),
        ("max_phase_deg", coarse.max_phase_deg / fine.max_phase_deg),
        ("mean_phase_deg", coarse.mean_phase_deg.abs() / fine.mean_phase_deg.abs()),
    ];
    for (name, ratio) in ratios {
        assert!(ratio > 3.0, "halving dx reduced {name} only {ratio:.2}x, expected ~4x");
    }
}
