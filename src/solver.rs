use nalgebra::DMatrix;
use num_complex::Complex;
use complex_bessel::hankel2;
use complex_bessel::besselj;
use rustfft::{FftDirection, FftPlanner};
use kryst::algebra::bridge::BridgeScratch;
use kryst::ops::klinop::KLinOp;
use kryst::ops::kpc::KPreconditioner;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::GmresSolver;

pub const CC: f64 = 299792458.0; // Speed of light in vacuum
pub const PI: f64 = std::f64::consts::PI;
pub const MU0: f64 = 4.0 * PI * 1e-7; // Permeability of free space
pub const EPSILON0: f64 = 1.0 / MU0 / CC / CC;

/// The discretized imaging domain: a grid of `nx` x `ny` cells spanning
/// `x_meters` x `y_meters`. Index (0, 0) is both the array origin and the
/// spatial origin (top-left corner of the domain).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain {
    pub nx: usize,
    pub ny: usize,
    pub x_meters: f64,
    pub y_meters: f64,
}

impl Domain {
    pub fn dx(&self) -> f64 {
        self.x_meters / self.nx as f64
    }

    pub fn dy(&self) -> f64 {
        self.y_meters / self.ny as f64
    }

    pub fn center(&self) -> (f64, f64) {
        (self.x_meters / 2.0, self.y_meters / 2.0)
    }

    pub fn cells(&self) -> usize {
        self.nx * self.ny
    }
}

/// Background medium and operating frequency, from which the wavenumber,
/// wavelength and angular frequency are derived.
#[derive(Clone, Copy, Debug)]
pub struct Medium {
    pub frequency_hz: f64,
    pub background_permittivity: Complex<f64>,
}

impl Medium {
    pub fn omega(&self) -> f64 {
        2.0 * PI * self.frequency_hz
    }

    pub fn lambda(&self) -> f64 {
        CC / self.frequency_hz
    }

    pub fn k_b(&self) -> Complex<f64> {
        Complex::new(self.omega(), 0.0) * (EPSILON0 * self.background_permittivity * MU0).sqrt()
    }
}

// nalgebra is column-major, so the columns form contiguous chunks that one process() call handles.
fn fft_columns(m: &mut DMatrix<Complex<f64>>, planner: &mut FftPlanner<f64>, dir: FftDirection) {
    let fft = planner.plan_fft(m.nrows(), dir);
    fft.process(m.as_mut_slice());
}

fn fft2_dir(input: &DMatrix<Complex<f64>>, dir: FftDirection) -> DMatrix<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let mut out = input.clone();
    fft_columns(&mut out, &mut planner, dir);
    let mut out_t = out.transpose();
    fft_columns(&mut out_t, &mut planner, dir);
    out_t.transpose()
}

pub fn fft2(input: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    fft2_dir(input, FftDirection::Forward)
}

// rustfft does not normalize, so scale by 1/(N*M) here.
pub fn ifft2(input: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    let scale = 1.0 / (input.nrows() * input.ncols()) as f64;
    fft2_dir(input, FftDirection::Inverse) * Complex::new(scale, 0.0)
}

pub fn contrast(eps: &DMatrix<Complex<f64>>, background: Complex<f64>) -> DMatrix<Complex<f64>> {
    eps.map(|val| (val - background) / background)
}

/// All pairwise cell-to-cell distances. O((nx*ny)^2) memory - only useful for
/// the direct (non-FFT) coefficient build below, not the API solve path.
pub fn rho(domain: &Domain) -> DMatrix<f64> {
    let (nx, ny) = (domain.nx, domain.ny);
    let (dx, dy) = (domain.dx(), domain.dy());
    let mut rho_matrix = DMatrix::<f64>::zeros(nx * ny, nx * ny);
    for m in 0..nx * ny {
        let (i_m, j_m) = (m / ny, m % ny);
        let x_m = dx * i_m as f64;
        let y_m = dy * j_m as f64;
        for n in 0..nx * ny {
            let (i_n, j_n) = (n / ny, n % ny);
            let x_n = dx * i_n as f64;
            let y_n = dy * j_n as f64;
            let distance = ((x_n - x_m).powi(2) + (y_n - y_m).powi(2)).sqrt();
            rho_matrix[(m, n)] = distance;
        }
    }
    rho_matrix
}

/// Dense direct-solve coefficient matrix (unused by the FFT/GMRES path; kept
/// for reference/benchmarking against the accelerated solver).
pub fn build_coeffs(
    domain: &Domain,
    a_out: &mut DMatrix<Complex<f64>>,
    contrast: &DMatrix<Complex<f64>>,
    k_b: Complex<f64>,
) {
    let (nx, ny) = (domain.nx, domain.ny);
    let a = (domain.dx() * domain.dy() / PI).sqrt(); // area of each cell
    let flattened_contrast: Vec<Complex<f64>> = contrast.transpose().iter().cloned().collect();
    let rho_matrix = rho(domain);
    let besselj_ka = besselj(1.0, k_b * a).unwrap();
    let hankel2_ka = hankel2(1.0, k_b * a).unwrap();
    for m in 0..nx * ny {
        for n in 0..nx * ny {
            if m == n {
                a_out[(m, n)] = Complex::new(1.0, 0.0)
                    + flattened_contrast[m]
                        * Complex::new(0.0, 1.0 / 2.0)
                        * (PI * k_b * a * hankel2_ka - Complex::new(0.0, 2.0))
            } else {
                a_out[(m, n)] = Complex::new(0.0, 1.0 * PI * a / 2.0)
                    * k_b
                    * flattened_contrast[n]
                    * besselj_ka
                    * hankel2(0.0, k_b * rho_matrix[(m, n)]).unwrap();
            }
        }
    }
}

pub fn build_kernel(domain: &Domain, k_b: Complex<f64>) -> DMatrix<Complex<f64>> {
    let (nx, ny) = (domain.nx, domain.ny);
    let (dx, dy) = (domain.dx(), domain.dy());
    let mut kernel = DMatrix::<Complex<f64>>::zeros(2 * nx - 1, 2 * ny - 1);
    let a = (dx * dy / PI).sqrt();
    let besselj_ka = besselj(1.0, k_b * a).unwrap();
    let hankel2_ka = hankel2(1.0, k_b * a).unwrap();
    for p in -(nx as i32 - 1)..=(nx as i32 - 1) {
        for q in -(ny as i32 - 1)..=(ny as i32 - 1) {
            let m = if p >= 0 { p } else { p + (2 * nx as i32 - 1) } as usize;
            let n = if q >= 0 { q } else { q + (2 * ny as i32 - 1) } as usize;
            let rho_pq = ((p as f64 * dx).powi(2) + (q as f64 * dy).powi(2)).sqrt();
            if p == 0 && q == 0 {
                kernel[(m, n)] = Complex::new(0.0, 1.0 / 2.0) * (PI * k_b * a * hankel2_ka - Complex::new(0.0, 2.0));
            } else {
                kernel[(m, n)] = Complex::new(0.0, 1.0 * PI * a / 2.0) * k_b * besselj_ka * hankel2(0.0, k_b * rho_pq).unwrap();
            }
        }
    }
    kernel
}

/// Fixed point iteration solve. Does not converge for large contrast - kept
/// for reference; the API path uses `gmres_solve` instead.
pub fn iter_solve(
    domain: &Domain,
    kernel: DMatrix<Complex<f64>>,
    contrast: DMatrix<Complex<f64>>,
    u_inc: DMatrix<Complex<f64>>,
    max_iter: usize,
    tol: f64,
) -> DMatrix<Complex<f64>> {
    let (nx, ny) = (domain.nx, domain.ny);
    let mut u_tot = u_inc.clone();
    let kernel_fft = fft2(&kernel);
    let kernel_nrows = kernel.nrows();
    let kernel_ncols = kernel.ncols();

    let mut contrast_src = DMatrix::<Complex<f64>>::zeros(kernel_nrows, kernel_ncols);

    for iter in 0..max_iter {
        let temp_contrast_src = &contrast.component_mul(&u_tot);
        for i in 0..nx {
            for j in 0..ny {
                contrast_src[(i, j)] = temp_contrast_src[(i, j)];
            }
        }
        let contrast_src_fft = fft2(&contrast_src);
        let field_fft = contrast_src_fft.component_mul(&kernel_fft);
        let padded_field = ifft2(&field_fft);
        let mut field = DMatrix::<Complex<f64>>::zeros(nx, ny);
        for i in 0..nx {
            for j in 0..ny {
                field[(i, j)] = padded_field[(i, j)];
            }
        }
        let residual = (&field + &u_tot - &u_inc).norm();
        u_tot = &u_inc - &field;

        if residual < tol {
            break;
        }
        let _ = iter;
    }
    u_tot
}

// Matrix-free operator A u = u + G * (contrast .* u), with G applied as an FFT convolution.
// Vectors are the column-major flattening of an nx x ny grid.
struct IntegralOp {
    nx: usize,
    ny: usize,
    kernel_fft: DMatrix<Complex<f64>>,
    contrast: DMatrix<Complex<f64>>,
}

impl KLinOp for IntegralOp {
    type Scalar = Complex<f64>;

    fn dims(&self) -> (usize, usize) {
        (self.nx * self.ny, self.nx * self.ny)
    }

    fn matvec_s(&self, x: &[Complex<f64>], y: &mut [Complex<f64>], _scratch: &mut BridgeScratch) {
        let u = DMatrix::from_column_slice(self.nx, self.ny, x);
        let mut padded = DMatrix::<Complex<f64>>::zeros(self.kernel_fft.nrows(), self.kernel_fft.ncols());
        padded.view_mut((0, 0), (self.nx, self.ny)).copy_from(&self.contrast.component_mul(&u));
        let conv = ifft2(&fft2(&padded).component_mul(&self.kernel_fft));
        for j in 0..self.ny {
            for i in 0..self.nx {
                y[j * self.nx + i] = u[(i, j)] + conv[(i, j)];
            }
        }
    }
}

pub fn gmres_solve(
    domain: &Domain,
    kernel: &DMatrix<Complex<f64>>,
    contrast: &DMatrix<Complex<f64>>,
    u_inc: &DMatrix<Complex<f64>>,
    restart: usize,
    max_iter: usize,
    tol: f64,
) -> Result<DMatrix<Complex<f64>>, String> {
    let op = IntegralOp {
        nx: domain.nx,
        ny: domain.ny,
        kernel_fft: fft2(kernel),
        contrast: contrast.clone(),
    };
    let b = u_inc.as_slice();
    let mut x = b.to_vec();
    let comm = UniverseComm::NoComm(NoComm);
    let pc: Option<&dyn KPreconditioner<Scalar = Complex<f64>>> = None;

    let mut solver = GmresSolver::new(restart, tol, max_iter);
    let stats = solver
        .solve(&op, pc, b, &mut x, PcSide::Left, &comm, None, None)
        .map_err(|e| format!("GMRES failed: {e:?}"))?;
    tracing::debug!(
        iterations = stats.iterations,
        reason = ?stats.reason,
        final_residual = stats.final_residual,
        "GMRES solve finished"
    );
    // kryst reports hitting max_iter as a stop reason, not an error, so check
    // it here rather than silently returning an unconverged field.
    if !stats.reason.is_converged() {
        return Err(format!(
            "GMRES did not converge: {:?} after {} iterations, residual {:.3e} (tol {tol:.1e})",
            stats.reason, stats.iterations, stats.final_residual
        ));
    }
    Ok(DMatrix::from_vec(domain.nx, domain.ny, x))
}

/// The incident illumination.
#[derive(Clone, Copy, Debug)]
pub enum IncidentWave {
    /// Unit-amplitude plane wave traveling at `angle_rad` (0 = +x direction),
    /// phase-referenced to the domain center.
    PlaneWave { angle_rad: f64 },
    /// Unit line source at `(x, y)`: the 2D Green's function
    /// `-j/4 * H0^(2)(k_b * rho)`, consistent with the scattered-field kernel.
    PointSource { x: f64, y: f64 },
}

/// Line-source field at `obs`. At the source itself the Green's function is
/// singular, so within `a` of it we use the average over a disk of radius `a`
/// (the same equivalent-circle cell model used for the kernel self-term).
fn point_source_at(k_b: Complex<f64>, src: (f64, f64), obs: (f64, f64), a: f64) -> Complex<f64> {
    let distance = ((obs.0 - src.0).powi(2) + (obs.1 - src.1).powi(2)).sqrt();
    if distance < a * 1e-6 {
        let ka = k_b * a;
        Complex::new(0.0, -0.5) * hankel2(1.0, ka).unwrap() / ka - 1.0 / (PI * ka * ka)
    } else {
        Complex::new(0.0, -0.25) * hankel2(0.0, k_b * distance).unwrap()
    }
}

pub fn inc_field(domain: &Domain, k_b: Complex<f64>, incident: IncidentWave) -> DMatrix<Complex<f64>> {
    let (dx, dy) = (domain.dx(), domain.dy());
    DMatrix::<Complex<f64>>::from_fn(domain.nx, domain.ny, |i, j| {
        inc_at_obs(domain, k_b, (dx * i as f64, dy * j as f64), incident)
    })
}

pub fn scat_at_obs(
    domain: &Domain,
    k_b: Complex<f64>,
    contrast: &DMatrix<Complex<f64>>,
    u_tot: &DMatrix<Complex<f64>>,
    obs: (f64, f64),
) -> Complex<f64> {
    let (nx, ny) = (domain.nx, domain.ny);
    let (dx, dy) = (domain.dx(), domain.dy());
    let mut scattered = Complex::new(0.0, 0.0);
    for i in 0..nx {
        for j in 0..ny {
            let x_i = dx * i as f64;
            let y_j = dy * j as f64;
            let distance = ((obs.0 - x_i).powi(2) + (obs.1 - y_j).powi(2)).sqrt();
            if distance > 0.0 && contrast[(i, j)].norm() > 0.0 {
                scattered += Complex::new(0.0, -1.0 / 4.0) * k_b * k_b * contrast[(i, j)] * u_tot[(i, j)] * hankel2(0.0, k_b * distance).unwrap() * dx * dy;
            }
        }
    }
    scattered
}

pub fn inc_at_obs(domain: &Domain, k_b: Complex<f64>, obs: (f64, f64), incident: IncidentWave) -> Complex<f64> {
    match incident {
        IncidentWave::PlaneWave { angle_rad } => {
            let (cx, cy) = domain.center();
            let phase = (obs.0 - cx) * angle_rad.cos() + (obs.1 - cy) * angle_rad.sin();
            (Complex::new(0.0, -1.0) * k_b * Complex::new(phase, 0.0)).exp()
        }
        IncidentWave::PointSource { x, y } => {
            let a = (domain.dx() * domain.dy() / PI).sqrt();
            point_source_at(k_b, (x, y), obs, a)
        }
    }
}

pub fn sample_tot(domain: &Domain, points: &[(f64, f64)], u_tot: &DMatrix<Complex<f64>>) -> Vec<Complex<f64>> {
    let (dx, dy) = (domain.dx(), domain.dy());
    let mut samples = Vec::new();
    for &(x, y) in points {
        let i = (x / dx).round() as usize;
        let j = (y / dy).round() as usize;
        if i < domain.nx && j < domain.ny {
            samples.push(u_tot[(i, j)]);
        } else {
            samples.push(Complex::new(0.0, 0.0));
        }
    }
    samples
}

pub fn receivers_circle(domain: &Domain, num_receivers: usize, radius: f64) -> Vec<(f64, f64)> {
    let (cx, cy) = domain.center();
    (0..num_receivers)
        .map(|n| {
            let angle = PI * n as f64 / num_receivers as f64; // half circle
            (cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect()
}

pub fn receivers_full_circle(domain: &Domain, num_receivers: usize, radius: f64) -> Vec<(f64, f64)> {
    let (cx, cy) = domain.center();
    (0..num_receivers)
        .map(|n| {
            let angle = 2.0 * PI * n as f64 / num_receivers as f64;
            (cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect()
}

pub fn receiver_data(
    domain: &Domain,
    k_b: Complex<f64>,
    contrast: &DMatrix<Complex<f64>>,
    u_tot: &DMatrix<Complex<f64>>,
    receivers: &[(f64, f64)],
) -> Vec<Complex<f64>> {
    receivers.iter().map(|&r| scat_at_obs(domain, k_b, contrast, u_tot, r)).collect()
}

pub fn inc_at_receivers(domain: &Domain, k_b: Complex<f64>, receivers: &[(f64, f64)], incident: IncidentWave) -> Vec<Complex<f64>> {
    receivers.iter().map(|&r| inc_at_obs(domain, k_b, r, incident)).collect()
}

pub fn echo_width(u_scat: &[Complex<f64>], u_inc: &[Complex<f64>], rho: f64, lambda: f64) -> Vec<f64> {
    u_scat
        .iter()
        .zip(u_inc.iter())
        .map(|(scattered, incident)| 2.0 * PI * rho * (scattered / incident).norm().powf(2.0) / lambda)
        .collect()
}

/// Parameters for a single forward-solve request.
pub struct SolveRequest {
    pub domain: Domain,
    pub medium: Medium,
    /// Absolute permittivity at every grid cell, `nx * ny` entries indexed
    /// `i * ny + j` (row-major over the (i, j) grid used throughout).
    pub permittivity: Vec<Complex<f64>>,
    pub incident: IncidentWave,
    pub receivers: Vec<(f64, f64)>,
    pub restart: usize,
    pub max_iter: usize,
    pub tol: f64,
}

pub struct SolveResult {
    pub total_field: DMatrix<Complex<f64>>,
    pub scattered_field: DMatrix<Complex<f64>>,
    pub receiver_scattered: Vec<Complex<f64>>,
    pub receiver_incident: Vec<Complex<f64>>,
}

pub fn solve(req: &SolveRequest) -> Result<SolveResult, String> {
    let domain = &req.domain;
    if req.permittivity.len() != domain.cells() {
        return Err(format!(
            "permittivity has {} entries, expected nx*ny = {}",
            req.permittivity.len(),
            domain.cells()
        ));
    }

    let eps = DMatrix::<Complex<f64>>::from_fn(domain.nx, domain.ny, |i, j| req.permittivity[i * domain.ny + j]);
    let contrast_matrix = contrast(&eps, req.medium.background_permittivity);
    let k_b = req.medium.k_b();

    let kernel = build_kernel(domain, k_b);
    let u_inc = inc_field(domain, k_b, req.incident);
    let u_tot = gmres_solve(domain, &kernel, &contrast_matrix, &u_inc, req.restart, req.max_iter, req.tol)?;
    let u_scat = &u_tot - &u_inc;

    let receiver_scattered = receiver_data(domain, k_b, &contrast_matrix, &u_tot, &req.receivers);
    let receiver_incident = inc_at_receivers(domain, k_b, &req.receivers, req.incident);

    Ok(SolveResult {
        total_field: u_tot,
        scattered_field: u_scat,
        receiver_scattered,
        receiver_incident,
    })
}
