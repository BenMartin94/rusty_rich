use nalgebra::DMatrix;
use num_complex::Complex;
use plotly::{Plot, HeatMap, Configuration};
use plotly::layout::{Axis, Layout};
use complex_bessel::hankel2;
use complex_bessel::besselj;
use std::time::Instant;
use faer::{Mat, c64};
use faer::prelude::Solve;
use rustfft::{FftDirection, FftPlanner};

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

const Nx: usize = 100;
const Ny: usize = 100;
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
const circle_permittivity: Complex<f64> = Complex::new(1.1, -0.05); // Permittivity of the circle

// top left corner of the domain index 0, 0 is the origin of the domain too. 

fn circle(eps: &mut DMatrix<Complex<f64>>, radius: f64, center: (f64, f64)) {
    let (cx, cy) = center;
    for i in 0..Nx {
        for j in 0..Ny {
            let x_i = dx * i as f64;
            let y_j = dy * j as f64;
            let distance = ((x_i - cx).powi(2) + (y_j - cy).powi(2)).sqrt();
            if distance <= radius {
                eps[(i, j)] = circle_permittivity; // Inside the circle
            } else {
                eps[(i, j)] = background_permittivity; // Outside the circle
            }
        }
    }
            
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

fn contrast(eps: &DMatrix<Complex<f64>>, background: Complex<f64>) -> DMatrix<Complex<f64>> {
    let contrast_matrix = eps.map(|val| (val - background)/background);
    contrast_matrix
}

fn rho()-> DMatrix<f64> {
    // for each grid point, compute the distance to every other grid point.
    let mut rho_matrix = DMatrix::<f64>::zeros(Nx*Ny, Nx*Ny);
    for m in 0..Nx*Ny {
        let (i_m, j_m) = (m / Nx, m % Nx);
        let x_m = dx * i_m as f64;
        let y_m = dy * j_m as f64;
        for n in 0..Nx*Ny {
            let (i_n, j_n) = (n / Nx, n % Nx);
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

fn inc_field(k_b: Complex<f64>) -> DMatrix<Complex<f64>> {
    let mut u_inc = DMatrix::<Complex<f64>>::zeros(Nx, Ny);
    for i in 0..Nx {
        for j in 0..Ny {
            let x_i = dx * i as f64;
            let y_j = dy * j as f64;
            u_inc[(i, j)] = Complex::new(1.0, 0.0) * (Complex::new(0.0, -1.0) * k_b * Complex::new(x_i, 0.0)).exp(); // plane wave traveling in the (1,0) direction
        }
    }
    u_inc
}



fn main() {
    
    let t_total = Instant::now();

    if lambda/10.0 < dx || lambda/10.0 < dy {
        println!("Warning: The grid spacing is too large for the wavelength. Consider reducing dx and dy.");
        println!("Current grid spacing: dx = {}, dy = {}, wavelength/10 = {}", dx, dy, lambda/10.0);
    }

    let mut eps = DMatrix::<Complex<f64>>::zeros(Ny, Nx);
    // create target grid now
    circle(&mut eps, 0.3*lambda, (x_meters/2.0, y_meters/2.0)); // Circle with radius 0.05 meters at center (0.15, 0.15)

    let contrast = contrast(&eps, background_permittivity);

    let k_b: Complex<f64> = Complex::new(omega, 0.0) * (epsilon0 * background_permittivity * mu0).sqrt(); // wave number in the background

    let u_inc = inc_field(k_b);
    plot_matrix(&u_inc.map(|c| c.re), "Incident field (real part)");

    // FFT-accelerated iterative solve
    println!("FFT solve...");
    let t_fft = Instant::now();
    let kernel = build_kernel(k_b);
    let t_kernel = t_fft.elapsed();
    let u_tot = iter_solve(kernel, contrast.clone(), u_inc.clone(), 100, 1e-6);
    let t_fft = t_fft.elapsed();
    println!("  build kernel: {:.3?}", t_kernel);
    println!("FFT solve total: {:.3?}", t_fft);

    // Direct N^2 build + LU solve
    let u_inc_flat = DMatrix::<Complex<f64>>::from_iterator(Nx*Ny, 1, u_inc.transpose().iter().cloned());
    let t_direct = Instant::now();
    let mut A = DMatrix::<Complex<f64>>::zeros(Nx*Ny, Nx*Ny);
    println!("Building coefficient matrix A...");
    let t_build = Instant::now();
    build_coeffs(&mut A, &contrast, k_b);
    println!("Build A: {:.3?}", t_build.elapsed());

    println!("Solving the linear system...");
    let t_solve = Instant::now();
    let a_faer = to_faer(&A);
    let b_faer = to_faer(&u_inc_flat);
    println!("  convert to faer: {:.3?}", t_solve.elapsed());
    let t_lu = Instant::now();
    let lu = a_faer.partial_piv_lu();
    println!("  lu factor: {:.3?}", t_lu.elapsed());
    let t_bs = Instant::now();
    let x_faer = lu.solve(b_faer.as_ref());
    println!("  back-substitution: {:.3?}", t_bs.elapsed());
    let u_tot_flat = DMatrix::<Complex<f64>>::from_iterator(
        Nx*Ny, 1,
        (0..Nx*Ny).map(|i| Complex::new(x_faer[(i, 0)].re, x_faer[(i, 0)].im))
    );
    println!("Solve: {:.3?}", t_solve.elapsed());
    let t_direct = t_direct.elapsed();
    println!("Direct solve total: {:.3?}", t_direct);
    let u_tot_direct = DMatrix::<Complex<f64>>::from_iterator(Ny, Nx, u_tot_flat.iter().cloned()).transpose();

    println!("----");
    println!("Grid {}x{} ({} unknowns)", Nx, Ny, Nx*Ny);
    println!("  FFT:    {:.3?}", t_fft);
    println!("  Direct: {:.3?}", t_direct);
    println!("  Speedup: {:.1}x", t_direct.as_secs_f64() / t_fft.as_secs_f64());
    println!("  Relative difference FFT vs direct: {:.3e}", (&u_tot - &u_tot_direct).norm() / u_tot_direct.norm());
    println!("Total (before plotting): {:.3?}", t_total.elapsed());

    plot_matrix(&u_tot.map(|c| c.re), "Total field, FFT solver (real part)");
    let uscattered = &u_tot - &u_inc;
    plot_matrix(&uscattered.map(|c| c.norm()), "Scattered field, FFT solver (magnitude)");

    plot_matrix(&u_tot_direct.map(|c| c.re), "Total field, direct solver (real part)");
    let uscattered_direct = &u_tot_direct - &u_inc;
    plot_matrix(&uscattered_direct.map(|c| c.norm()), "Scattered field, direct solver (magnitude)");
}
