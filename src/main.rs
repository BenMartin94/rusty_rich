use nalgebra::DMatrix;
use num_complex::Complex;
use plotly::{Plot, HeatMap, Configuration, Scatter};
use plotly::common::Mode;
use plotly::layout::{Axis, Layout};
use complex_bessel::hankel2;
use complex_bessel::besselj;
use std::time::Instant;
use faer::{Mat, c64};
use faer::prelude::Solve;
use rustfft::{FftDirection, FftPlanner};
use kryst::algebra::bridge::BridgeScratch;
use kryst::ops::klinop::KLinOp;
use kryst::ops::kpc::KPreconditioner;
use kryst::parallel::{NoComm, UniverseComm};
use kryst::preconditioner::PcSide;
use kryst::solver::GmresSolver;

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

fn fft2(input: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    fft2_dir(input, FftDirection::Forward)
}

// rustfft does not normalize, so scale by 1/(N*M) here.
fn ifft2(input: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    let scale = 1.0 / (input.nrows() * input.ncols()) as f64;
    fft2_dir(input, FftDirection::Inverse) * Complex::new(scale, 0.0)
}

fn to_faer(m: &DMatrix<Complex<f64>>) -> Mat<c64> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| c64::new(m[(i, j)].re, m[(i, j)].im))
}

const Nx: usize = 1001;
const Ny: usize = 1001;
const x_meters: f64 = 1.0;
const y_meters: f64 = 1.0;
const dx: f64 = x_meters / Nx as f64;
const dy: f64 = y_meters / Ny as f64;

const CC: f64 = 299792458.0; // Speed of light in vacuum
const frequency: f64 = CC;
const pi : f64 = std::f64::consts::PI;
const mu0: f64 = 4.0 * pi * 1e-7; // Permeability of free space
const epsilon0: f64 = 1.0/mu0/CC/CC;
const omega: f64 = 2.0 * pi * frequency; // Angular frequency
const lambda: f64 = CC / frequency; // Wavelength

const background_permittivity: Complex<f64> = Complex::new(1.0, 0.0); // Permittivity of the background
const richy_permittivity: Complex<f64> = Complex::new(4.0, 0.0); // Permittivity of the circle
const colin_permittivity: Complex<f64> = Complex::new(1.1, -0.05); // Permittivity of the colin target

// top left corner of the domain index 0, 0 is the origin of the domain too. 

fn circle(eps: &mut DMatrix<Complex<f64>>, radius: f64, center: (f64, f64)) {
    let (cx, cy) = center;
    for i in 0..Nx {
        for j in 0..Ny {
            let x_i = dx * i as f64;
            let y_j = dy * j as f64;
            let distance = ((x_i - cx).powi(2) + (y_j - cy).powi(2)).sqrt();
            if distance <= radius {
                eps[(i, j)] = colin_permittivity; // Inside the circle
            } else {
                eps[(i, j)] = background_permittivity; // Outside the circle
            }
        }
    }
            
}

fn richys_target(inner_radius: f64, outer_radius: f64) -> DMatrix<Complex<f64>> {
    // create a region of permittivity within the inner radius and outer radius. 
    let mut eps = DMatrix::<Complex<f64>>::zeros(Nx, Ny);
    for i in 0..Nx {
        for j in 0..Ny {
            let x_i = dx * i as f64;
            let y_j = dy * j as f64;
            let distance = ((x_i - x_meters/2.0).powi(2) + (y_j - y_meters/2.0).powi(2)).sqrt();
            if distance <= outer_radius && distance >= inner_radius {
                eps[(i, j)] = richy_permittivity; // Inside 
            } else {
                eps[(i, j)] = background_permittivity; // Outside 
            }
        }
    }
    eps
}

fn plot_matrix(matrix: &DMatrix<f64>, title: &str) {
    // plotly puts z rows on the y-axis, but our matrices index (x, y)
    let z: Vec<Vec<f64>> = (0..matrix.ncols())
        .map(|j| (0..matrix.nrows()).map(|i| matrix[(i, j)]).collect())
        .collect();

    let heatmap = HeatMap::new_z(z);
    let mut plot = Plot::new();
    plot.add_trace(heatmap);

    let layout = Layout::new()
        .title(title)
        .x_axis(Axis::new().constrain(plotly::layout::AxisConstrain::Domain))
        .y_axis(Axis::new().scale_anchor("x").scale_ratio(1.0))
        .auto_size(true);
    plot.set_layout(layout);
    plot.set_configuration(Configuration::new().responsive(true));

    plot.show();
}

fn plot_line(x: &[f64], y: &[f64], title: &str, x_label: &str, y_label: &str) {
    let trace = Scatter::new(x.to_vec(), y.to_vec()).mode(Mode::LinesMarkers);
    let mut plot = Plot::new();
    plot.add_trace(trace);

    let layout = Layout::new()
        .title(title)
        .x_axis(Axis::new().title(x_label))
        .y_axis(Axis::new().title(y_label))
        .auto_size(true);
    plot.set_layout(layout);
    plot.set_configuration(Configuration::new().responsive(true));

    plot.show();
}

fn contrast(eps: &DMatrix<Complex<f64>>, background: Complex<f64>) -> DMatrix<Complex<f64>> {
    let contrast_matrix = eps.map(|val| (val - background)/background);
    contrast_matrix
}

fn rho()-> DMatrix<f64> {
    // for each grid point, compute the distance to every other grid point.
    let mut rho_matrix = DMatrix::<f64>::zeros(Nx*Ny, Nx*Ny);
    for m in 0..Nx*Ny {
        let (i_m, j_m) = (m / Ny, m % Ny);
        let x_m = dx * i_m as f64;
        let y_m = dy * j_m as f64;
        for n in 0..Nx*Ny {
            let (i_n, j_n) = (n / Ny, n % Ny);
            let x_n = dx * i_n as f64;
            let y_n = dy * j_n as f64;
            let distance = ((x_n - x_m).powi(2) + (y_n - y_m).powi(2)).sqrt();
            rho_matrix[(m, n)] = distance;
        }
    }
    rho_matrix
}

fn build_coeffs(A: &mut DMatrix<Complex<f64>>, contrast: &DMatrix<Complex<f64>>, k_b: Complex<f64>) {
    //let k = eps.map(|e| Complex::new(omega, 0.0) * (e * mu0).sqrt()); // wave number in the medium
    // all cells are the same size so we can precompute the area
    let a = (dx*dy/pi).sqrt(); // area of each cell
    let flattened_contrast: Vec<Complex<f64>> = contrast.transpose().iter().cloned().collect();
    let rho_matrix = rho();
    let besselj_ka = besselj(1.0, k_b*a).unwrap();
    let hankel2_ka = hankel2(1.0, k_b*a).unwrap();
    for m in 0..Nx*Ny {
        for n in 0..Nx*Ny {

            if m == n {
                A[(m, n)] = Complex::new(1.0, 0.0) + flattened_contrast[m] * Complex::new(0.0, 1.0/2.0) * (pi*k_b*a*hankel2_ka-Complex::new(0.0, 2.0))
            }
            else{

                A[(m, n)] = Complex::new(0.0, 1.0*pi*a/2.0)*k_b*flattened_contrast[n]*besselj_ka*hankel2(0.0, k_b*rho_matrix[(m, n)]).unwrap();
            }

        }
    }

} 

fn build_kernel(k_b: Complex<f64>) -> DMatrix<Complex<f64>> {
    let mut kernel = DMatrix::<Complex<f64>>::zeros(2*Nx-1, 2*Ny-1);
    let a = (dx*dy/pi).sqrt();
    let besselj_ka = besselj(1.0, k_b*a).unwrap();
    let hankel2_ka = hankel2(1.0, k_b*a).unwrap();
    for p in -(Nx as i32 - 1)..=(Nx as i32 - 1){
        for q in -(Ny as i32 - 1)..=(Ny as i32 - 1){
            let m = if p >= 0 { p } else { p + (2*Nx as i32 - 1) } as usize;
            let n = if q >= 0 { q } else { q + (2*Ny as i32 - 1) } as usize;
            let rho_pq = ((p as f64 * dx).powi(2) + (q as f64 * dy).powi(2)).sqrt();
            if p == 0 && q == 0 {
                kernel[(m, n)] = Complex::new(0.0, 1.0/2.0) * (pi*k_b*a*hankel2_ka-Complex::new(0.0, 2.0));
            } else {
                kernel[(m, n)] = Complex::new(0.0, 1.0*pi*a/2.0)*k_b*besselj_ka*hankel2(0.0, k_b*rho_pq).unwrap();
            }
        }
    }

    // print the kernel
    kernel
}

fn iter_solve(kernel: DMatrix<Complex<f64>>, contrast: DMatrix<Complex<f64>>, u_inc: DMatrix<Complex<f64>>, max_iter: usize, tol: f64) -> DMatrix<Complex<f64>> {
    // my fixed point iteration iterative solve. Does not converge for large contrast. For large contrast use the GMRES solver.
    let mut u_tot = u_inc.clone();
    let mut kernel_fft = fft2(&kernel);
    let kernel_nrows = kernel.nrows();
    let kernel_ncols = kernel.ncols();

    let mut contrast_src = DMatrix::<Complex<f64>>::zeros(kernel_nrows, kernel_ncols); // allocated space for my contrast sources

    let mut residual = f64::MAX;
    for iter in 0..max_iter {
        // compute the contrast source
        let temp_contrast_src = &contrast.component_mul(&u_tot);
        // put temp_contrast_src into the top left corner of contrast_src
        for i in 0..Nx {
            for j in 0..Ny {
                contrast_src[(i, j)] = temp_contrast_src[(i, j)];
            }
        }
        // fft the contrast source
        let contrast_src_fft = fft2(&contrast_src);
        // element-wise multiply with the kernel in the frequency domain
        let field_fft = contrast_src_fft.component_mul(&kernel_fft);
        // inverse fft to get the scattered field in the spatial domain
        let padded_field = ifft2(&field_fft);
        // extract the top left Nx x Ny part of the scattered field
        let mut field = DMatrix::<Complex<f64>>::zeros(Nx, Ny);
        for i in 0..Nx {
            for j in 0..Ny {
                field[(i, j)] = padded_field[(i, j)];
            }
        }
        residual = (&field+&u_tot-&u_inc).norm();
        u_tot = &u_inc - &field; // update the total field
        
        println!("Iteration {}: residual = {}", iter, residual);
        if residual < tol {
            println!("Converged after {} iterations with residual {}", iter, residual);
            break;
        }
    }
    u_tot
}

// Matrix-free operator A u = u + G * (contrast .* u), with G applied as an FFT convolution.
// Vectors are the column-major flattening of an Nx x Ny grid.
struct IntegralOp {
    kernel_fft: DMatrix<Complex<f64>>,
    contrast: DMatrix<Complex<f64>>,
}

impl KLinOp for IntegralOp {
    type Scalar = Complex<f64>;

    fn dims(&self) -> (usize, usize) {
        (Nx * Ny, Nx * Ny)
    }

    fn matvec_s(&self, x: &[Complex<f64>], y: &mut [Complex<f64>], _scratch: &mut BridgeScratch) {
        // Ax = u + G * (contrast .* u)
        // where x and u are synonymous.
        // This function computes y = Ax for a given x, using FFTs to compute the convolution with G.
        let u = DMatrix::from_column_slice(Nx, Ny, x);
        let mut padded = DMatrix::<Complex<f64>>::zeros(self.kernel_fft.nrows(), self.kernel_fft.ncols());
        padded.view_mut((0, 0), (Nx, Ny)).copy_from(&self.contrast.component_mul(&u));
        let conv = ifft2(&fft2(&padded).component_mul(&self.kernel_fft));
        for j in 0..Ny {
            for i in 0..Nx {
                y[j * Nx + i] = u[(i, j)] + conv[(i, j)];
            }
        }
    }
}

fn gmres_solve(kernel: &DMatrix<Complex<f64>>, contrast: &DMatrix<Complex<f64>>, u_inc: &DMatrix<Complex<f64>>, restart: usize, max_iter: usize, tol: f64) -> DMatrix<Complex<f64>> {
    let op = IntegralOp {
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
        .expect("GMRES failed");
    println!(
        "GMRES: {} iterations, reason {:?}, final residual {:.3e}",
        stats.iterations, stats.reason, stats.final_residual
    );
    DMatrix::from_vec(Nx, Ny, x)
}

fn inc_field(k_b: Complex<f64>) -> DMatrix<Complex<f64>> {
    let mut u_inc = DMatrix::<Complex<f64>>::zeros(Nx, Ny);
    for i in 0..Nx {
        for j in 0..Ny {
            let x_i = dx * i as f64;
            let y_j = dy * j as f64;
            u_inc[(i, j)] = Complex::new(1.0, 0.0) * (Complex::new(0.0, -1.0) * k_b * Complex::new(x_i - x_meters/2.0, 0.0)).exp(); // plane wave traveling in the (1,0) direction
        }
    }
    u_inc
}

fn scat_at_obs(k_b: Complex<f64>, contrast: &DMatrix<Complex<f64>>, u_tot: &DMatrix<Complex<f64>>, obs: (f64, f64)) -> Complex<f64> {
    let mut scattered = Complex::new(0.0, 0.0);
    for i in 0..Nx {
        for j in 0..Ny {
            let x_i = dx * i as f64;
            let y_j = dy * j as f64;
            let distance = ((obs.0 - x_i).powi(2) + (obs.1 - y_j).powi(2)).sqrt();
            if distance > 0.0 && contrast[(i, j)].norm() > 0.0 {
                scattered += Complex::new(0.0, -1.0/4.0) *k_b*k_b * contrast[(i, j)] * u_tot[(i, j)] * hankel2(0.0, k_b*distance).unwrap() * dx * dy;
            }
        }
    }
    scattered
}

fn inc_at_obs(k_b: Complex<f64>, obs: (f64, f64)) -> Complex<f64> {
    let x = obs.0;
    let y = obs.1;
    Complex::new(1.0, 0.0) * (Complex::new(0.0, -1.0) * k_b * Complex::new(x-x_meters/2.0, 0.0)).exp() // plane wave traveling in the (1,0) direction
}

fn sample_tot(points: &Vec<(f64, f64)>, u_tot: &DMatrix<Complex<f64>>) -> Vec<Complex<f64>> {
    let mut samples = Vec::new();
    for &(x, y) in points {
        let i = (x / dx).round() as usize;
        let j = (y / dy).round() as usize;
        if i < Nx && j < Ny {
            samples.push(u_tot[(i, j)]);
        } else {
            samples.push(Complex::new(0.0, 0.0)); // out of bounds
        }
    }
    samples
}

fn receivers_circle(num_receivers: usize, radius: f64) -> Vec<(f64, f64)> {
    let mut receivers = Vec::new();
    for n in 0..num_receivers {
        let angle = std::f64::consts::PI * n as f64 / num_receivers as f64; // just cover a half circle
        let x = x_meters/2.0 + radius * angle.cos();
        let y = y_meters/2.0 + radius * angle.sin();
        receivers.push((x, y));
    }
    receivers
}

fn receivers_full_circle(num_receivers: usize, radius: f64) -> Vec<(f64, f64)> {
    let mut receivers = Vec::new();
    for n in 0..num_receivers {
        let angle = 2.0 * std::f64::consts::PI * n as f64 / num_receivers as f64; // full circle
        let x = x_meters/2.0 + radius * angle.cos();
        let y = y_meters/2.0 + radius * angle.sin();
        receivers.push((x, y));
    }
    receivers
}

fn receiver_data(k_b: Complex<f64>, contrast: &DMatrix<Complex<f64>>, u_tot: &DMatrix<Complex<f64>>, receivers: &Vec<(f64, f64)>) -> Vec<Complex<f64>> {
    let mut data = Vec::new();
    for &receiver in receivers {
        let scattered = scat_at_obs(k_b, contrast, u_tot, receiver);
        data.push(scattered);
    }
    data
}

fn inc_at_receivers(k_b: Complex<f64>, receivers: &Vec<(f64, f64)>) -> Vec<Complex<f64>> {
    let mut data = Vec::new();
    for &receiver in receivers {
        let incident = inc_at_obs(k_b, receiver);
        data.push(incident);
    }
    data
}

fn echo_width(u_scat: &Vec<Complex<f64>>, u_inc: &Vec<Complex<f64>>, rho: f64)->Vec<f64> {
    let mut echo_widths = Vec::new();
    for phi in 0..u_scat.len() {
        let scattered = u_scat[phi];
        let incident = u_inc[phi];
        let echo_width = 2.0 * pi * rho * (scattered/incident).norm().powf(2.0)/lambda; // echo width formula
        echo_widths.push(echo_width);
    }
    echo_widths
}

fn main() {
    
    let t_total = Instant::now();

    if lambda/10.0 < dx || lambda/10.0 < dy {
        println!("Warning: The grid spacing is too large for the wavelength. Consider reducing dx and dy.");
        println!("Current grid spacing: dx = {}, dy = {}, wavelength/10 = {}", dx, dy, lambda/10.0);
    }

    let mut eps = DMatrix::<Complex<f64>>::zeros(Nx, Ny);
    // create target grid now
    circle(&mut eps, 0.3*lambda, (x_meters/2.0, y_meters/2.0)); // Circle with radius 0.05 meters at center (0.15, 0.15)
    let richys_eps = richys_target(0.25*lambda, 0.3*lambda);
    plot_matrix(&richys_eps.map(|c| c.re), "Richy's target permittivity (real part)");
    let rich_contrast = contrast(&richys_eps, background_permittivity);

    let k_b: Complex<f64> = Complex::new(omega, 0.0) * (epsilon0 * background_permittivity * mu0).sqrt(); // wave number in the background

    let u_inc = inc_field(k_b);
    plot_matrix(&u_inc.map(|c| c.re), "Incident field (real part)");

    // FFT-accelerated iterative solve
    println!("FFT solve...");
    let t_fft = Instant::now();
    let kernel = build_kernel(k_b);
    let t_kernel = t_fft.elapsed();
    let u_tot = gmres_solve(&kernel, &rich_contrast, &u_inc, 30, 500, 1e-6);
    let t_fft = t_fft.elapsed();
    println!("  build kernel: {:.3?}", t_kernel);
    println!("FFT solve total: {:.3?}", t_fft);

    //|u| at the shell
    let num_receivers = 128;

    let phi: Vec<f64> = (0..num_receivers).map(|n| 180.0 * n as f64 / num_receivers as f64).collect();
    let full_circle_phi: Vec<f64> = (0..num_receivers).map(|n| 360.0 * n as f64 / num_receivers as f64).collect();
    let shell_points = receivers_circle(128, 0.275*lambda);
    let u_tot_shell = sample_tot(&shell_points, &u_tot);
    let u_tot_shell_mag: Vec<f64> = u_tot_shell.iter().map(|c| c.norm()).collect();
    plot_line(&phi, &u_tot_shell_mag, "Figure 3 Recreated", "phi (degrees)", "|E|");

    // Compute scattered field at receivers
    // find the echo width now
    let receivers = receivers_circle(num_receivers, 2.5*lambda);
    let scattered_data = receiver_data(k_b, &rich_contrast, &u_tot, &receivers);
    let incident_data = inc_at_receivers(k_b, &receivers);
    let echo_widths = echo_width(&scattered_data, &incident_data, 2.5*lambda);
    plot_line(&phi, &echo_widths, "Figure 4 Recreated", "phi (degrees)", "Echo width/lambda");

    // time for colins one
    let colin_contrast = contrast(&eps, background_permittivity);
    let colin_u_tot = gmres_solve(&kernel, &colin_contrast, &u_inc, 30, 500, 1e-6);

    let colin_receivers = receivers_full_circle(num_receivers, 0.5*lambda);
    let colin_scattered_data = receiver_data(k_b, &colin_contrast, &colin_u_tot, &colin_receivers);
    plot_line(&full_circle_phi, &colin_scattered_data.iter().map(|c| c.norm()).collect::<Vec<f64>>(), "Colin's target scattered field", "observation angle", "|E_sct|");
    plot_line(&full_circle_phi, &colin_scattered_data.iter().map(|c| c.arg()).collect::<Vec<f64>>(), "Colin's target scattered field", "observation angle", "E_sct phase");


    
    

    // Direct N^2 build + LU solve
    // let u_inc_flat = DMatrix::<Complex<f64>>::from_iterator(Nx*Ny, 1, u_inc.transpose().iter().cloned());
    // let t_direct = Instant::now();
    // let mut A = DMatrix::<Complex<f64>>::zeros(Nx*Ny, Nx*Ny);
    // println!("Building coefficient matrix A...");
    // let t_build = Instant::now();
    // build_coeffs(&mut A, &contrast, k_b);
    // println!("Build A: {:.3?}", t_build.elapsed());

    // println!("Solving the linear system...");
    // let t_solve = Instant::now();
    // let a_faer = to_faer(&A);
    // let b_faer = to_faer(&u_inc_flat);
    // println!("  convert to faer: {:.3?}", t_solve.elapsed());
    // let t_lu = Instant::now();
    // let lu = a_faer.partial_piv_lu();
    // println!("  lu factor: {:.3?}", t_lu.elapsed());
    // let t_bs = Instant::now();
    // let x_faer = lu.solve(b_faer.as_ref());
    // println!("  back-substitution: {:.3?}", t_bs.elapsed());
    // let u_tot_flat = DMatrix::<Complex<f64>>::from_iterator(
    //     Nx*Ny, 1,
    //     (0..Nx*Ny).map(|i| Complex::new(x_faer[(i, 0)].re, x_faer[(i, 0)].im))
    // );
    // println!("Solve: {:.3?}", t_solve.elapsed());
    // let t_direct = t_direct.elapsed();
    // println!("Direct solve total: {:.3?}", t_direct);
    // let u_tot_direct = DMatrix::<Complex<f64>>::from_iterator(Ny, Nx, u_tot_flat.iter().cloned()).transpose();

    // println!("----");
    // println!("Grid {}x{} ({} unknowns)", Nx, Ny, Nx*Ny);
    // println!("  FFT:    {:.3?}", t_fft);
    // println!("  Direct: {:.3?}", t_direct);
    // println!("  Speedup: {:.1}x", t_direct.as_secs_f64() / t_fft.as_secs_f64());
    // println!("  Relative difference FFT vs direct: {:.3e}", (&u_tot - &u_tot_direct).norm() / u_tot_direct.norm());
    // println!("Total (before plotting): {:.3?}", t_total.elapsed());

    plot_matrix(&u_tot.map(|c| c.re), "Total field, FFT solver (real part)");
    let uscattered = &u_tot - &u_inc;
    plot_matrix(&uscattered.map(|c| c.norm()), "Scattered field, FFT solver (magnitude)");

    // plot_matrix(&u_tot_direct.map(|c| c.re), "Total field, direct solver (real part)");
    // let uscattered_direct = &u_tot_direct - &u_inc;
    // plot_matrix(&uscattered_direct.map(|c| c.norm()), "Scattered field, direct solver (magnitude)");
}
