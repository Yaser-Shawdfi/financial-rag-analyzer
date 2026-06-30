use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{Duration as ChronoDuration, Utc};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::{Deserialize, Serialize};

use crate::config::AuthConfig;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: i64,
    pub iat: i64,
}

#[derive(Clone)]
pub struct AuthService {
    config: AuthConfig,
    rate_limiter: Arc<tokio::sync::RwLock<HashMap<String, (u32, Instant)>>>,
}

impl AuthService {
    pub fn new(config: AuthConfig) -> Self {
        Self {
            config,
            rate_limiter: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub fn issue_token(&self, username: &str) -> anyhow::Result<String> {
        let now = Utc::now();
        let exp = now + ChronoDuration::hours(self.config.jwt_expiry_hours);

        let claims = Claims {
            sub: username.to_string(),
            exp: exp.timestamp(),
            iat: now.timestamp(),
        };

        let header = serde_json::json!({"alg": "HS256", "typ": "JWT"});
        let header_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header)?);
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims)?);

        let signing_input = format!("{}.{}", header_b64, payload_b64);

        let mut mac = HmacSha256::new_from_slice(self.config.jwt_secret.as_bytes())?;
        mac.update(signing_input.as_bytes());
        let signature = mac.finalize().into_bytes();
        let sig_b64 = URL_SAFE_NO_PAD.encode(signature);

        Ok(format!("{}.{}", signing_input, sig_b64))
    }

    pub fn verify_token(&self, token: &str) -> Option<Claims> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return None;
        }

        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let expected_sig = URL_SAFE_NO_PAD.decode(parts[2]).ok()?;

        let mut mac = HmacSha256::new_from_slice(self.config.jwt_secret.as_bytes()).ok()?;
        mac.update(signing_input.as_bytes());

        if mac.verify_slice(&expected_sig).is_err() {
            return None;
        }

        let payload = URL_SAFE_NO_PAD.decode(parts[1]).ok()?;
        let claims: Claims = serde_json::from_slice(&payload).ok()?;

        if Utc::now().timestamp() > claims.exp {
            return None;
        }

        Some(claims)
    }

    pub async fn check_rate_limit(&self, username: &str) -> bool {
        let mut limiter = self.rate_limiter.write().await;
        let now = Instant::now();
        let window = Duration::from_secs(60);

        let entry = limiter.entry(username.to_string()).or_insert((0, now));

        if now.duration_since(entry.1) > window {
            *entry = (0, now);
        }

        entry.0 += 1;
        entry.0 <= self.config.rate_limit_per_minute
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }
}

pub async fn auth_middleware(
    State(auth): State<Arc<AuthService>>,
    mut req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, (StatusCode, String)> {
    if !auth.enabled() {
        return Ok(next.run(req).await);
    }

    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or((StatusCode::UNAUTHORIZED, "Missing Authorization header".into()))?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or((StatusCode::UNAUTHORIZED, "Invalid auth scheme".into()))?;

    let claims = auth
        .verify_token(token)
        .ok_or((StatusCode::UNAUTHORIZED, "Invalid or expired token".into()))?;

    if !auth.check_rate_limit(&claims.sub).await {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded".into()));
    }

    req.extensions_mut().insert(claims.sub);

    Ok(next.run(req).await)
}

use axum::extract::State;