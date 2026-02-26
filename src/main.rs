// Rust Text-to-Speech Starter - Backend Server
//
// This is a simple Rust HTTP server that provides a text-to-speech API endpoint
// powered by Deepgram's Text-to-Speech service. It's designed to be easily
// modified and extended for your own projects.
//
// Key Features:
// - Contract-compliant API endpoint: POST /api/text-to-speech
// - Accepts text in body and model as query parameter
// - Returns binary audio data (audio/mpeg)
// - JWT session auth for API protection
// - CORS enabled for frontend communication
// - Pure API server (frontend served separately)

use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderMap, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, fs, sync::Arc};
use tower_http::cors::{Any, CorsLayer};

// ============================================================================
// CONFIGURATION - Customize these values for your needs
// ============================================================================

/// Default text-to-speech model to use when none is specified.
/// Options: "aura-2-thalia-en", "aura-2-theia-en", "aura-2-andromeda-en", etc.
/// See: https://developers.deepgram.com/docs/text-to-speech-models
const DEFAULT_MODEL: &str = "aura-2-thalia-en";

/// JWT token expiry duration in seconds (1 hour).
const JWT_EXPIRY_SECS: i64 = 3600;

// ============================================================================
// TYPES - Structures for request/response handling
// ============================================================================

/// TTSRequest represents the JSON body for the text-to-speech endpoint.
#[derive(Deserialize)]
struct TTSRequest {
    text: Option<String>,
}

/// ErrorDetail holds structured error information matching the contract format.
#[derive(Serialize)]
struct ErrorDetail {
    r#type: String,
    code: String,
    message: String,
    details: HashMap<String, String>,
}

/// ErrorResponse wraps an ErrorDetail in the contract-compliant structure.
#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

/// SessionResponse holds the JWT token issued by /api/session.
#[derive(Serialize)]
struct SessionResponse {
    token: String,
}

/// HealthResponse holds the health check response.
#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

/// Query parameters for the text-to-speech endpoint.
#[derive(Deserialize)]
struct TTSQueryParams {
    model: Option<String>,
}

/// JWT claims for session tokens.
#[derive(Serialize, Deserialize)]
struct Claims {
    iat: i64,
    exp: i64,
}

/// DeepgramToml represents the parsed deepgram.toml file.
#[derive(Deserialize)]
struct DeepgramToml {
    meta: Option<toml::Value>,
}

/// Shared application state passed to route handlers.
#[derive(Clone)]
struct AppState {
    api_key: String,
    session_secret: String,
}

// ============================================================================
// SESSION AUTH - JWT tokens for API protection
// ============================================================================

/// Generate a random hex string of the given byte length.
fn generate_random_hex(n: usize) -> String {
    let mut bytes = vec![0u8; n];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

/// Initialize the session secret from env or generate a random one.
fn init_session_secret() -> String {
    match env::var("SESSION_SECRET") {
        Ok(secret) if !secret.is_empty() => secret,
        _ => generate_random_hex(32),
    }
}

/// Create a signed JWT with the configured session secret.
fn create_jwt(secret: &str) -> Result<String, jsonwebtoken::errors::Error> {
    let now = Utc::now().timestamp();
    let claims = Claims {
        iat: now,
        exp: now + JWT_EXPIRY_SECS,
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
}

/// Validate a JWT token string and return an error if invalid.
fn verify_jwt(token_string: &str, secret: &str) -> Result<(), String> {
    let validation = Validation::default();
    decode::<Claims>(
        token_string,
        &DecodingKey::from_secret(secret.as_bytes()),
        &validation,
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

// ============================================================================
// API KEY LOADING - Load Deepgram API key from environment
// ============================================================================

/// Read the Deepgram API key from environment variables.
/// Exits with a helpful error message if not found.
fn load_api_key() -> String {
    match env::var("DEEPGRAM_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            eprintln!();
            eprintln!("ERROR: Deepgram API key not found!");
            eprintln!();
            eprintln!("Please set your API key using one of these methods:");
            eprintln!();
            eprintln!("1. Create a .env file (recommended):");
            eprintln!("   DEEPGRAM_API_KEY=your_api_key_here");
            eprintln!();
            eprintln!("2. Environment variable:");
            eprintln!("   export DEEPGRAM_API_KEY=your_api_key_here");
            eprintln!();
            eprintln!("Get your API key at: https://console.deepgram.com");
            eprintln!();
            std::process::exit(1);
        }
    }
}

// ============================================================================
// AUTH MIDDLEWARE - JWT Bearer token validation
// ============================================================================

/// Middleware that validates JWT Bearer tokens on protected routes.
/// Returns 401 with structured error if token is missing or invalid.
async fn require_auth(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let auth_header = req.headers().get(header::AUTHORIZATION);

    let token_string = match auth_header.and_then(|v| v.to_str().ok()) {
        Some(auth) if auth.starts_with("Bearer ") => &auth[7..],
        _ => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        r#type: "AuthenticationError".to_string(),
                        code: "MISSING_TOKEN".to_string(),
                        message: "Authorization header with Bearer token is required".to_string(),
                        details: HashMap::new(),
                    },
                }),
            )
                .into_response();
        }
    };

    match verify_jwt(token_string, &state.session_secret) {
        Ok(()) => next.run(req).await,
        Err(err) => {
            let message = if err.to_lowercase().contains("expired") {
                "Session expired, please refresh the page"
            } else {
                "Invalid session token"
            };
            (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        r#type: "AuthenticationError".to_string(),
                        code: "INVALID_TOKEN".to_string(),
                        message: message.to_string(),
                        details: HashMap::new(),
                    },
                }),
            )
                .into_response()
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS - Modular logic for easier understanding and testing
// ============================================================================

/// Build a contract-compliant error response.
/// Auto-detects the error code from the message if not explicitly provided.
fn format_error_response(
    message: &str,
    status_code: u16,
    error_code: Option<&str>,
) -> ErrorResponse {
    let code = match error_code {
        Some(c) => c.to_string(),
        None => {
            let msg_lower = message.to_lowercase();
            if status_code == 400 {
                if msg_lower.contains("empty") {
                    "EMPTY_TEXT".to_string()
                } else if msg_lower.contains("model") {
                    "MODEL_NOT_FOUND".to_string()
                } else if msg_lower.contains("long")
                    || msg_lower.contains("limit")
                    || msg_lower.contains("exceed")
                {
                    "TEXT_TOO_LONG".to_string()
                } else {
                    "INVALID_TEXT".to_string()
                }
            } else {
                "INVALID_TEXT".to_string()
            }
        }
    };

    let error_type = if status_code == 400 {
        "ValidationError"
    } else {
        "GenerationError"
    };

    let mut details = HashMap::new();
    details.insert("originalError".to_string(), message.to_string());

    ErrorResponse {
        error: ErrorDetail {
            r#type: error_type.to_string(),
            code,
            message: message.to_string(),
            details,
        },
    }
}

// ============================================================================
// DEEPGRAM API - Direct HTTP calls to the Deepgram TTS endpoint
// ============================================================================

/// Call the Deepgram TTS API directly and return the audio bytes.
/// Sends a JSON body with the text and passes the model as a query parameter.
async fn generate_audio(api_key: &str, text: &str, model: &str) -> Result<Vec<u8>, String> {
    let client = reqwest::Client::new();

    let url = format!("https://api.deepgram.com/v1/speak?model={}", model);

    let mut payload = HashMap::new();
    payload.insert("text", text);

    let resp = client
        .post(&url)
        .header("Authorization", format!("Token {}", api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("failed to call Deepgram API: {}", e))?;

    let status = resp.status();
    let body = resp
        .bytes()
        .await
        .map_err(|e| format!("failed to read response: {}", e))?;

    if !status.is_success() {
        let body_str = String::from_utf8_lossy(&body);
        return Err(format!(
            "Deepgram API error (status {}): {}",
            status.as_u16(),
            body_str
        ));
    }

    Ok(body.to_vec())
}

// ============================================================================
// ROUTE HANDLERS - API endpoint implementations
// ============================================================================

/// Issue a signed JWT for session authentication.
/// GET /api/session
async fn handle_session(State(state): State<Arc<AppState>>) -> Response {
    match create_jwt(&state.session_secret) {
        Ok(token) => Json(SessionResponse { token }).into_response(),
        Err(e) => {
            eprintln!("Failed to create JWT: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Failed to create session"})),
            )
                .into_response()
        }
    }
}

/// Convert text to speech audio via the Deepgram API.
/// POST /api/text-to-speech?model=aura-2-thalia-en
///
/// Accepts JSON body: {"text": "Hello world"}
/// Returns binary audio data (audio/mpeg) on success.
async fn handle_text_to_speech(
    State(state): State<Arc<AppState>>,
    Query(params): Query<TTSQueryParams>,
    Json(body): Json<TTSRequest>,
) -> Response {
    let model = params.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());

    // Validate input - text is required
    let text = match &body.text {
        Some(t) if !t.is_empty() => t.clone(),
        Some(t) if t.is_empty() => {
            let err = format_error_response("Text parameter is required", 400, Some("EMPTY_TEXT"));
            return (StatusCode::BAD_REQUEST, Json(err)).into_response();
        }
        _ => {
            let err = format_error_response("Text parameter is required", 400, Some("EMPTY_TEXT"));
            return (StatusCode::BAD_REQUEST, Json(err)).into_response();
        }
    };

    if text.trim().is_empty() {
        let err =
            format_error_response("Text must be a non-empty string", 400, Some("EMPTY_TEXT"));
        return (StatusCode::BAD_REQUEST, Json(err)).into_response();
    }

    // Generate audio from text via Deepgram API
    match generate_audio(&state.api_key, &text, &model).await {
        Ok(audio_data) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, "audio/mpeg".parse().unwrap());
            (StatusCode::OK, headers, audio_data).into_response()
        }
        Err(err) => {
            eprintln!("Text-to-speech error: {}", err);
            let err_lower = err.to_lowercase();

            let (status_code, error_code) =
                if err_lower.contains("model") || err_lower.contains("not found") {
                    (400u16, Some("MODEL_NOT_FOUND"))
                } else if err_lower.contains("too long")
                    || err_lower.contains("length")
                    || err_lower.contains("limit")
                    || err_lower.contains("exceed")
                {
                    (400, Some("TEXT_TOO_LONG"))
                } else if err_lower.contains("invalid") || err_lower.contains("malformed") {
                    (400, Some("INVALID_TEXT"))
                } else {
                    (500, None)
                };

            let err_resp = format_error_response(&err, status_code, error_code);
            let http_status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (http_status, Json(err_resp)).into_response()
        }
    }
}

/// Return project metadata from deepgram.toml.
/// GET /api/metadata
async fn handle_metadata() -> Response {
    let content = match fs::read_to_string("deepgram.toml") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading deepgram.toml: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "Failed to read metadata from deepgram.toml"
                })),
            )
                .into_response();
        }
    };

    let config: DeepgramToml = match toml::from_str(&content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error parsing deepgram.toml: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "Failed to read metadata from deepgram.toml"
                })),
            )
                .into_response();
        }
    };

    match config.meta {
        Some(meta) => Json(meta).into_response(),
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "INTERNAL_SERVER_ERROR",
                "message": "Missing [meta] section in deepgram.toml"
            })),
        )
            .into_response(),
    }
}

/// Return a simple health check response.
/// GET /health
async fn handle_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

// ============================================================================
// SERVER START
// ============================================================================

#[tokio::main]
async fn main() {
    // Load .env file (ignore error if not present)
    let _ = dotenvy::dotenv();

    // Load API key and initialize session
    let api_key = load_api_key();
    let session_secret = init_session_secret();

    // Read port and host from environment
    let port = env::var("PORT").unwrap_or_else(|_| "8081".to_string());
    let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());

    let state = Arc::new(AppState {
        api_key,
        session_secret,
    });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    // Build the router with unprotected and protected routes
    let app = Router::new()
        // Unprotected routes
        .route("/api/session", get(handle_session))
        .route("/api/metadata", get(handle_metadata))
        .route("/health", get(handle_health))
        // Protected routes (auth required)
        .route(
            "/api/text-to-speech",
            post(handle_text_to_speech).route_layer(middleware::from_fn_with_state(
                state.clone(),
                require_auth,
            )),
        )
        .layer(cors)
        .with_state(state);

    let addr = format!("{}:{}", host, port);

    println!();
    println!("{}", "=".repeat(70));
    println!("Backend API running at http://localhost:{}", port);
    println!("GET  /api/session");
    println!("POST /api/text-to-speech (auth required)");
    println!("GET  /api/metadata");
    println!("GET  /health");
    println!("{}", "=".repeat(70));
    println!();

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Failed to bind to {}: {}", addr, e);
            std::process::exit(1);
        });

    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Server error: {}", e);
            std::process::exit(1);
        });
}
