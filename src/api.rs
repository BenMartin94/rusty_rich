use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;
use axum::http::{header, HeaderValue};
use uuid::Uuid;

use crate::solver::{self, Domain, Medium};

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct ComplexDto {
    pub re: f64,
    pub im: f64,
}

impl From<ComplexDto> for Complex<f64> {
    fn from(c: ComplexDto) -> Self {
        Complex::new(c.re, c.im)
    }
}

impl From<Complex<f64>> for ComplexDto {
    fn from(c: Complex<f64>) -> Self {
        ComplexDto { re: c.re, im: c.im }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct DomainDto {
    pub nx: usize,
    pub ny: usize,
    pub x_meters: f64,
    pub y_meters: f64,
}

impl From<DomainDto> for Domain {
    fn from(d: DomainDto) -> Self {
        Domain { nx: d.nx, ny: d.ny, x_meters: d.x_meters, y_meters: d.y_meters }
    }
}

impl From<Domain> for DomainDto {
    fn from(d: Domain) -> Self {
        DomainDto { nx: d.nx, ny: d.ny, x_meters: d.x_meters, y_meters: d.y_meters }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct PointDto {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Deserialize, Clone, Copy, Default)]
pub struct SolverParamsDto {
    pub restart: Option<usize>,
    pub max_iter: Option<usize>,
    pub tol: Option<f64>,
}

/// A forward-solve request: the imaging domain, background medium, the
/// permittivity grid drawn/uploaded by the client, and the receiver points
/// to sample the resulting field at.
#[derive(Debug, Deserialize, Clone)]
pub struct SolveRequestDto {
    pub domain: DomainDto,
    pub frequency_hz: f64,
    pub background_permittivity: ComplexDto,
    /// Absolute permittivity at every grid cell, flattened row-major as
    /// `i * ny + j`, length `nx * ny`.
    pub permittivity_re: Vec<f64>,
    pub permittivity_im: Vec<f64>,
    #[serde(default)]
    pub incident_angle_deg: f64,
    /// If set, illuminate with a line source here instead of a plane wave
    /// (`incident_angle_deg` is then ignored).
    pub source_position: Option<PointDto>,
    pub receivers: Vec<PointDto>,
    pub solver: Option<SolverParamsDto>,
}

#[derive(Debug, Serialize, Clone)]
pub struct FieldGridDto {
    pub nx: usize,
    pub ny: usize,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

fn flatten_field(m: &nalgebra::DMatrix<Complex<f64>>) -> FieldGridDto {
    let nx = m.nrows();
    let ny = m.ncols();
    let mut re = Vec::with_capacity(nx * ny);
    let mut im = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            re.push(m[(i, j)].re);
            im.push(m[(i, j)].im);
        }
    }
    FieldGridDto { nx, ny, re, im }
}

#[derive(Debug, Serialize, Clone)]
pub struct ReceiverResultDto {
    pub x: f64,
    pub y: f64,
    pub scattered: ComplexDto,
    pub incident: ComplexDto,
    /// `incident + scattered`.
    pub total: ComplexDto,
}

fn receiver_results(receivers: &[(f64, f64)], result: &solver::SolveResult) -> Vec<ReceiverResultDto> {
    receivers
        .iter()
        .zip(result.receiver_scattered.iter())
        .zip(result.receiver_incident.iter())
        .map(|((&(x, y), &scattered), &incident)| ReceiverResultDto {
            x,
            y,
            scattered: scattered.into(),
            incident: incident.into(),
            total: (scattered + incident).into(),
        })
        .collect()
}

#[derive(Debug, Serialize, Clone)]
pub struct SolveResponseDto {
    pub domain: DomainDto,
    pub total_field: FieldGridDto,
    pub scattered_field: FieldGridDto,
    pub receivers: Vec<ReceiverResultDto>,
    pub solve_time_ms: u128,
}

pub enum ApiError {
    Validation(String),
    Internal(String),
    NotFound(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::Validation(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
        };
        (status, message).into_response()
    }
}

fn build_incident(incident_angle_deg: f64, source_position: Option<PointDto>) -> solver::IncidentWave {
    match source_position {
        Some(p) => solver::IncidentWave::PointSource { x: p.x, y: p.y },
        None => solver::IncidentWave::PlaneWave { angle_rad: incident_angle_deg.to_radians() },
    }
}

/// Validates the domain and permittivity grid shared by all solve requests.
fn build_domain_and_permittivity(
    domain: DomainDto,
    permittivity_re: &[f64],
    permittivity_im: &[f64],
) -> Result<(Domain, Vec<Complex<f64>>), ApiError> {
    let domain: Domain = domain.into();
    if domain.nx == 0 || domain.ny == 0 {
        return Err(ApiError::Validation("domain.nx and domain.ny must be > 0".into()));
    }
    if permittivity_re.len() != domain.cells() || permittivity_im.len() != domain.cells() {
        return Err(ApiError::Validation(format!(
            "permittivity_re/permittivity_im must have nx*ny = {} entries",
            domain.cells()
        )));
    }
    let permittivity = permittivity_re
        .iter()
        .zip(permittivity_im.iter())
        .map(|(&re, &im)| Complex::new(re, im))
        .collect();
    Ok((domain, permittivity))
}

/// Validates the incoming request shape and converts it into the internal
/// solver request. Cheap and synchronous - run before ever spawning compute.
fn build_solve_request(req: &SolveRequestDto) -> Result<solver::SolveRequest, ApiError> {
    let (domain, permittivity) = build_domain_and_permittivity(req.domain, &req.permittivity_re, &req.permittivity_im)?;
    let receivers: Vec<(f64, f64)> = req.receivers.iter().map(|p| (p.x, p.y)).collect();
    let solver_params = req.solver.unwrap_or_default();

    Ok(solver::SolveRequest {
        domain,
        medium: Medium {
            frequency_hz: req.frequency_hz,
            background_permittivity: req.background_permittivity.into(),
        },
        permittivity,
        incident: build_incident(req.incident_angle_deg, req.source_position),
        receivers,
        restart: solver_params.restart.unwrap_or(30),
        max_iter: solver_params.max_iter.unwrap_or(500),
        tol: solver_params.tol.unwrap_or(1e-6),
    })
}

fn run_solve(solve_req: solver::SolveRequest) -> Result<SolveResponseDto, String> {
    let domain = solve_req.domain;
    let start = std::time::Instant::now();
    let result = solver::solve(&solve_req)?;
    let solve_time_ms = start.elapsed().as_millis();

    Ok(SolveResponseDto {
        domain: domain.into(),
        total_field: flatten_field(&result.total_field),
        scattered_field: flatten_field(&result.scattered_field),
        receivers: receiver_results(&solve_req.receivers, &result),
        solve_time_ms,
    })
}

// ---- multi-frequency endpoint (API only, not used by the frontend) ----

/// One target permittivity grid solved at several frequencies. Only the
/// receiver fields are returned, not the full grids.
#[derive(Debug, Deserialize, Clone)]
pub struct MultiFreqSolveRequestDto {
    pub domain: DomainDto,
    pub frequencies_hz: Vec<f64>,
    /// Treated as frequency-independent, like the target permittivity.
    pub background_permittivity: ComplexDto,
    /// Absolute permittivity at every grid cell, flattened row-major as
    /// `i * ny + j`, length `nx * ny`.
    pub permittivity_re: Vec<f64>,
    pub permittivity_im: Vec<f64>,
    #[serde(default)]
    pub incident_angle_deg: f64,
    /// If set, illuminate with a line source here instead of a plane wave
    /// (`incident_angle_deg` is then ignored).
    pub source_position: Option<PointDto>,
    pub receivers: Vec<PointDto>,
    pub solver: Option<SolverParamsDto>,
}

#[derive(Debug, Serialize, Clone)]
pub struct FrequencyResultDto {
    pub frequency_hz: f64,
    pub receivers: Vec<ReceiverResultDto>,
    pub solve_time_ms: u128,
}

#[derive(Debug, Serialize, Clone)]
pub struct MultiFreqSolveResponseDto {
    pub domain: DomainDto,
    /// In the same order as the requested `frequencies_hz`.
    pub results: Vec<FrequencyResultDto>,
    pub solve_time_ms: u128,
}

fn build_multifreq_requests(req: &MultiFreqSolveRequestDto) -> Result<Vec<solver::SolveRequest>, ApiError> {
    if req.frequencies_hz.is_empty() {
        return Err(ApiError::Validation("frequencies_hz must not be empty".into()));
    }
    if let Some(f) = req.frequencies_hz.iter().find(|f| !f.is_finite() || **f <= 0.0) {
        return Err(ApiError::Validation(format!("invalid frequency {f}: must be finite and > 0")));
    }
    let (domain, permittivity) = build_domain_and_permittivity(req.domain, &req.permittivity_re, &req.permittivity_im)?;
    let receivers: Vec<(f64, f64)> = req.receivers.iter().map(|p| (p.x, p.y)).collect();
    let solver_params = req.solver.unwrap_or_default();

    Ok(req
        .frequencies_hz
        .iter()
        .map(|&frequency_hz| solver::SolveRequest {
            domain,
            medium: Medium {
                frequency_hz,
                background_permittivity: req.background_permittivity.into(),
            },
            permittivity: permittivity.clone(),
            incident: build_incident(req.incident_angle_deg, req.source_position),
            receivers: receivers.clone(),
            restart: solver_params.restart.unwrap_or(30),
            max_iter: solver_params.max_iter.unwrap_or(500),
            tol: solver_params.tol.unwrap_or(1e-6),
        })
        .collect())
}

fn run_receivers_only_solve(solve_req: solver::SolveRequest) -> Result<FrequencyResultDto, String> {
    let start = std::time::Instant::now();
    let result = solver::solve(&solve_req)?;
    Ok(FrequencyResultDto {
        frequency_hz: solve_req.medium.frequency_hz,
        receivers: receiver_results(&solve_req.receivers, &result),
        solve_time_ms: start.elapsed().as_millis(),
    })
}

/// Server-wide cap on concurrent multi-frequency solves, one per core.
static SOLVE_PERMITS: std::sync::LazyLock<Arc<tokio::sync::Semaphore>> = std::sync::LazyLock::new(|| {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    Arc::new(tokio::sync::Semaphore::new(cores))
});

async fn multifreq_solve_handler(
    axum::Json(req): axum::Json<MultiFreqSolveRequestDto>,
) -> Result<axum::Json<MultiFreqSolveResponseDto>, ApiError> {
    let solve_reqs = build_multifreq_requests(&req)?;
    let start = std::time::Instant::now();

    // Frequencies are independent, so solve them concurrently on the blocking
    // pool. The cap is shared across requests so several concurrent long
    // frequency lists don't oversubscribe the CPU.
    let handles: Vec<_> = solve_reqs
        .into_iter()
        .map(|solve_req| {
            let permits = SOLVE_PERMITS.clone();
            tokio::spawn(async move {
                let _permit = permits.acquire_owned().await.expect("semaphore closed");
                tokio::task::spawn_blocking(move || run_receivers_only_solve(solve_req)).await
            })
        })
        .collect();

    let mut results = Vec::with_capacity(handles.len());
    for (handle, &frequency_hz) in handles.into_iter().zip(req.frequencies_hz.iter()) {
        let outcome = handle
            .await
            .map_err(|e| ApiError::Internal(format!("solver task panicked: {e}")))?
            .map_err(|e| ApiError::Internal(format!("solver task panicked: {e}")))?
            .map_err(|e| ApiError::Validation(format!("solve at {frequency_hz} Hz failed: {e}")))?;
        results.push(outcome);
    }

    Ok(axum::Json(MultiFreqSolveResponseDto {
        domain: req.domain,
        results,
        solve_time_ms: start.elapsed().as_millis(),
    }))
}

// ---- synchronous endpoint (handy for quick tests / small grids) ----

async fn solve_handler(axum::Json(req): axum::Json<SolveRequestDto>) -> Result<axum::Json<SolveResponseDto>, ApiError> {
    let solve_req = build_solve_request(&req)?;
    // The physics solve is CPU-bound and synchronous; run it on the blocking
    // pool so it doesn't stall the async runtime handling other requests.
    let result = tokio::task::spawn_blocking(move || run_solve(solve_req))
        .await
        .map_err(|e| ApiError::Internal(format!("solver task panicked: {e}")))?
        .map_err(ApiError::Validation)?;
    Ok(axum::Json(result))
}

// ---- job-based endpoints ----
//
// Solves on realistic grids can run for seconds to minutes, so the primary
// flow is: POST /api/jobs to enqueue, then poll GET /api/jobs/:id until it
// reports done/failed. Jobs live in memory only and are lost on restart.

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "status", rename_all = "snake_case")]
enum JobStatusDto {
    Pending,
    Running,
    Done(SolveResponseDto),
    Failed { error: String },
}

#[derive(Clone, Default)]
struct JobsStore(Arc<Mutex<HashMap<Uuid, JobStatusDto>>>);

impl JobsStore {
    fn insert(&self, id: Uuid, status: JobStatusDto) {
        self.0.lock().unwrap().insert(id, status);
    }

    fn get(&self, id: &Uuid) -> Option<JobStatusDto> {
        self.0.lock().unwrap().get(id).cloned()
    }
}

#[derive(Clone, Default)]
pub struct AppState {
    jobs: JobsStore,
}

#[derive(Debug, Serialize)]
struct CreateJobResponseDto {
    job_id: String,
}

async fn create_job_handler(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<SolveRequestDto>,
) -> Result<axum::Json<CreateJobResponseDto>, ApiError> {
    let solve_req = build_solve_request(&req)?;

    let job_id = Uuid::new_v4();
    state.jobs.insert(job_id, JobStatusDto::Pending);

    let jobs = state.jobs.clone();
    tokio::spawn(async move {
        jobs.insert(job_id, JobStatusDto::Running);
        let outcome = tokio::task::spawn_blocking(move || run_solve(solve_req)).await;
        let status = match outcome {
            Ok(Ok(response)) => JobStatusDto::Done(response),
            Ok(Err(message)) => JobStatusDto::Failed { error: message },
            Err(join_err) => JobStatusDto::Failed { error: format!("solver task panicked: {join_err}") },
        };
        jobs.insert(job_id, status);
    });

    Ok(axum::Json(CreateJobResponseDto { job_id: job_id.to_string() }))
}

async fn get_job_handler(State(state): State<AppState>, Path(job_id): Path<String>) -> Result<Response, ApiError> {
    let job_id = Uuid::parse_str(&job_id).map_err(|_| ApiError::Validation("invalid job id".into()))?;
    let status = state.jobs.get(&job_id).ok_or_else(|| ApiError::NotFound("job not found".into()))?;
    Ok(axum::Json(status).into_response())
}

async fn health_handler() -> &'static str {
    "ok"
}

pub fn app() -> Router {
    let state = AppState::default();

    Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/solve", post(solve_handler))
        .route("/api/solve/multifreq", post(multifreq_solve_handler))
        .route("/api/jobs", post(create_job_handler))
        .route("/api/jobs/{job_id}", get(get_job_handler))
        .with_state(state)
        .layer(CorsLayer::permissive())
        .fallback_service(ServeDir::new("frontend"))
        // Without this, browsers heuristically cache app.js and keep running
        // stale frontend code after edits; no-cache forces a cheap ETag revalidation.
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-cache"),
        ))
}
