use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use ring::rand::{SecureRandom, SystemRandom};
use rusqlite::{params, Connection, OptionalExtension};

const MAX_CLIENT_NAME_BYTES: usize = 128;
const MAX_REDIRECT_URIS: usize = 10;
const MAX_REDIRECT_URI_BYTES: usize = 2048;
const MAX_OAUTH_CLIENTS: i64 = 256;

#[derive(Debug, Clone)]
pub struct RegisterClientInput {
    pub client_name: String,
    pub redirect_uris: Vec<String>,
    pub grant_types: Vec<String>,
    pub token_endpoint_auth_method: String,
}

#[derive(Debug, Clone)]
pub struct RegisteredClient {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub client_name: String,
    pub redirect_uris: Vec<String>,
    pub grant_types: Vec<String>,
    pub token_endpoint_auth_method: String,
    pub client_id_issued_at: i64,
    pub client_secret_expires_at: i64,
}

#[derive(Debug, Clone)]
pub struct OAuthClient {
    pub client_id: String,
    pub client_secret_hash: Option<String>,
    pub client_name: String,
    pub redirect_uris: Vec<String>,
    pub grant_types: Vec<String>,
    pub token_endpoint_auth_method: String,
    pub registered_at: i64,
    pub last_used_at: Option<i64>,
    pub revoked_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct InsertAuthCodeInput {
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub expires_at: i64,
}

#[derive(Debug, Clone)]
pub struct ConsumedAuthCode {
    pub code: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub expires_at: i64,
}

#[derive(Debug, Clone)]
pub struct InsertGrantInput {
    pub client_id: String,
    pub access_token_jti: String,
    pub access_expires_at: i64,
    pub refresh_token: String,
    pub refresh_expires_at: i64,
}

#[derive(Debug, Clone)]
pub struct GrantRecord {
    pub grant_id: String,
    pub client_id: String,
    pub access_token_jti: String,
    pub access_expires_at: i64,
    pub refresh_expires_at: i64,
    pub revoked_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub enum RefreshGrantLookup {
    Active(GrantRecord),
    ReusedOrExpired,
    NotFound,
}

#[derive(Debug, Clone)]
pub struct SigningKey {
    pub kid: String,
    pub secret_hex: String,
    pub created_at: i64,
    pub rotated_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ClientSummary {
    pub client_id: String,
    pub client_name: String,
    pub registered_at: i64,
    pub last_used_at: Option<i64>,
    pub revoked_at: Option<i64>,
    pub active_grants: i64,
}

pub fn random_token(bytes: usize) -> anyhow::Result<String> {
    let mut buf = vec![0u8; bytes];
    SystemRandom::new()
        .fill(&mut buf)
        .map_err(|_| anyhow::anyhow!("secure random generation failed"))?;
    Ok(URL_SAFE_NO_PAD.encode(buf))
}

pub fn hash_secret(secret: &str) -> anyhow::Result<String> {
    let mut salt = [0u8; 16];
    SystemRandom::new()
        .fill(&mut salt)
        .map_err(|_| anyhow::anyhow!("secure random generation failed"))?;
    let salt = SaltString::encode_b64(&salt)
        .map_err(|err| anyhow::anyhow!("argon2 salt encode failed: {err}"))?;
    Ok(Argon2::default()
        .hash_password(secret.as_bytes(), &salt)
        .map_err(|err| anyhow::anyhow!("argon2 hash failed: {err}"))?
        .to_string())
}

pub fn verify_secret(secret: &str, hash: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(hash) else {
        return false;
    };
    Argon2::default()
        .verify_password(secret.as_bytes(), &parsed)
        .is_ok()
}

fn validate_redirect_uri(uri: &str) -> anyhow::Result<()> {
    if uri.contains('#') {
        anyhow::bail!("redirect_uri must not contain a fragment");
    }
    let parsed = uri
        .parse::<hyper::Uri>()
        .map_err(|_| anyhow::anyhow!("invalid redirect_uri"))?;
    let scheme = parsed.scheme_str().unwrap_or("");
    let host = parsed.host().unwrap_or("");
    if scheme == "https" {
        return Ok(());
    }
    let normalized_host = host.trim_start_matches('[').trim_end_matches(']');
    let is_loopback_host = normalized_host == "localhost"
        || normalized_host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    if scheme == "http" && is_loopback_host {
        return Ok(());
    }
    anyhow::bail!("redirect_uri must use https or http localhost")
}

fn normalize_grant_types(grant_types: &[String]) -> anyhow::Result<Vec<String>> {
    let mut has_authorization_code = false;
    let mut has_refresh_token = false;
    for grant in grant_types {
        match grant.as_str() {
            "authorization_code" => has_authorization_code = true,
            "refresh_token" => has_refresh_token = true,
            _ => anyhow::bail!(
                "grant_types must contain authorization_code and optional refresh_token"
            ),
        }
    }
    if !has_authorization_code {
        anyhow::bail!("grant_types must contain authorization_code and optional refresh_token");
    }
    let mut normalized = vec!["authorization_code".to_string()];
    if has_refresh_token {
        normalized.push("refresh_token".to_string());
    }
    Ok(normalized)
}

fn validate_register_input(input: &RegisterClientInput) -> anyhow::Result<()> {
    let client_name = input.client_name.trim();
    if client_name.is_empty() {
        anyhow::bail!("client_name is required");
    }
    if client_name.len() > MAX_CLIENT_NAME_BYTES {
        anyhow::bail!("client_name exceeds {MAX_CLIENT_NAME_BYTES} bytes");
    }
    if input.redirect_uris.is_empty() {
        anyhow::bail!("redirect_uris is required");
    }
    if input.redirect_uris.len() > MAX_REDIRECT_URIS {
        anyhow::bail!("redirect_uris exceeds {MAX_REDIRECT_URIS} entries");
    }
    for uri in &input.redirect_uris {
        if uri.len() > MAX_REDIRECT_URI_BYTES {
            anyhow::bail!("redirect_uri exceeds {MAX_REDIRECT_URI_BYTES} bytes");
        }
        validate_redirect_uri(uri)?;
    }
    normalize_grant_types(&input.grant_types)?;
    if !matches!(
        input.token_endpoint_auth_method.as_str(),
        "none" | "client_secret_basic"
    ) {
        anyhow::bail!("unsupported token_endpoint_auth_method");
    }
    Ok(())
}

fn oauth_client_count(conn: &Connection) -> anyhow::Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM oauth_clients", [], |row| row.get(0))
        .map_err(Into::into)
}

fn oldest_revoked_client_ids(conn: &Connection, limit: i64) -> anyhow::Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT client_id
         FROM oauth_clients
         WHERE revoked_at IS NOT NULL
         ORDER BY revoked_at ASC, registered_at ASC, client_id ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |row| row.get(0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn oldest_unused_client_ids(conn: &Connection, limit: i64) -> anyhow::Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT c.client_id
         FROM oauth_clients c
         WHERE c.revoked_at IS NULL
           AND c.last_used_at IS NULL
           AND NOT EXISTS (
               SELECT 1 FROM oauth_auth_codes ac WHERE ac.client_id = c.client_id
           )
           AND NOT EXISTS (
               SELECT 1 FROM oauth_grants g WHERE g.client_id = c.client_id
           )
         ORDER BY c.registered_at ASC, c.client_id ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |row| row.get(0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn delete_oauth_clients(conn: &Connection, client_ids: &[String]) -> anyhow::Result<usize> {
    let mut deleted = 0;
    for client_id in client_ids {
        conn.execute(
            "DELETE FROM oauth_auth_codes WHERE client_id = ?1",
            [client_id],
        )?;
        conn.execute("DELETE FROM oauth_grants WHERE client_id = ?1", [client_id])?;
        deleted += conn.execute(
            "DELETE FROM oauth_clients WHERE client_id = ?1",
            [client_id],
        )?;
    }
    Ok(deleted)
}

fn prune_oauth_clients_to_cap(conn: &Connection, keep: i64) -> anyhow::Result<usize> {
    let mut client_count = oauth_client_count(conn)?;
    if client_count <= keep {
        return Ok(0);
    }
    let mut deleted = 0;

    let revoked = oldest_revoked_client_ids(conn, client_count - keep)?;
    deleted += delete_oauth_clients(conn, &revoked)?;
    client_count = oauth_client_count(conn)?;
    if client_count <= keep {
        return Ok(deleted);
    }

    let unused = oldest_unused_client_ids(conn, client_count - keep)?;
    deleted += delete_oauth_clients(conn, &unused)?;
    Ok(deleted)
}

pub fn register_client(
    conn: &Connection,
    input: RegisterClientInput,
) -> anyhow::Result<RegisteredClient> {
    validate_register_input(&input)?;
    let grant_types = normalize_grant_types(&input.grant_types)?;
    let client_id = random_token(32)?;
    let client_secret = (input.token_endpoint_auth_method != "none")
        .then(|| random_token(32))
        .transpose()?;
    let client_secret_hash = client_secret.as_deref().map(hash_secret).transpose()?;
    let now = chrono::Utc::now().timestamp();
    let redirect_uris_json = serde_json::to_string(&input.redirect_uris)?;
    let grant_types_json = serde_json::to_string(&grant_types)?;
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let registered = (|| -> anyhow::Result<RegisteredClient> {
        prune_oauth_clients_to_cap(conn, MAX_OAUTH_CLIENTS - 1)?;
        if oauth_client_count(conn)? >= MAX_OAUTH_CLIENTS {
            anyhow::bail!(
                "oauth client registration limit reached; revoke unused clients in Connectors"
            );
        }
        conn.execute(
            "INSERT INTO oauth_clients (
                client_id, client_secret_hash, client_name, redirect_uris, grant_types,
                token_endpoint_auth_method, registered_at, last_used_at, revoked_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, NULL)",
            params![
                client_id,
                client_secret_hash,
                input.client_name.trim(),
                redirect_uris_json,
                grant_types_json,
                input.token_endpoint_auth_method,
                now
            ],
        )?;
        Ok(RegisteredClient {
            client_id,
            client_secret,
            client_name: input.client_name.trim().to_string(),
            redirect_uris: input.redirect_uris,
            grant_types,
            token_endpoint_auth_method: input.token_endpoint_auth_method,
            client_id_issued_at: now,
            client_secret_expires_at: 0,
        })
    })();
    match registered {
        Ok(registered) => {
            conn.execute_batch("COMMIT")?;
            Ok(registered)
        }
        Err(err) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(err)
        }
    }
}

pub fn get_client(conn: &Connection, client_id: &str) -> anyhow::Result<Option<OAuthClient>> {
    conn.query_row(
        "SELECT client_id, client_secret_hash, client_name, redirect_uris, grant_types,
                token_endpoint_auth_method, registered_at, last_used_at, revoked_at
         FROM oauth_clients WHERE client_id = ?1 AND revoked_at IS NULL",
        [client_id],
        |row| {
            let redirect_uris_json: String = row.get(3)?;
            let grant_types_json: String = row.get(4)?;
            Ok(OAuthClient {
                client_id: row.get(0)?,
                client_secret_hash: row.get(1)?,
                client_name: row.get(2)?,
                redirect_uris: serde_json::from_str(&redirect_uris_json).unwrap_or_default(),
                grant_types: serde_json::from_str(&grant_types_json).unwrap_or_default(),
                token_endpoint_auth_method: row.get(5)?,
                registered_at: row.get(6)?,
                last_used_at: row.get(7)?,
                revoked_at: row.get(8)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn insert_auth_code(conn: &Connection, input: InsertAuthCodeInput) -> anyhow::Result<String> {
    let code = random_token(32)?;
    let now = chrono::Utc::now().timestamp();
    conn.execute(
        "INSERT INTO oauth_auth_codes (
            code, client_id, redirect_uri, code_challenge, code_challenge_method,
            issued_at, expires_at, consumed_at
         ) VALUES (?1, ?2, ?3, ?4, 'S256', ?5, ?6, NULL)",
        params![
            code,
            input.client_id,
            input.redirect_uri,
            input.code_challenge,
            now,
            input.expires_at
        ],
    )?;
    Ok(code)
}

pub fn consume_auth_code(
    conn: &Connection,
    code: &str,
    client_id: &str,
    redirect_uri: &str,
    now: i64,
) -> anyhow::Result<ConsumedAuthCode> {
    let row = conn
        .query_row(
            "SELECT code, client_id, redirect_uri, code_challenge, expires_at, consumed_at
             FROM oauth_auth_codes WHERE code = ?1",
            [code],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| anyhow::anyhow!("invalid authorization code"))?;
    if row.5.is_some() {
        anyhow::bail!("authorization code already consumed");
    }
    if row.4 <= now {
        anyhow::bail!("authorization code expired");
    }
    if row.1 != client_id || row.2 != redirect_uri {
        anyhow::bail!("authorization code client or redirect mismatch");
    }
    let changed = conn.execute(
        "UPDATE oauth_auth_codes SET consumed_at = ?1
         WHERE code = ?2 AND consumed_at IS NULL",
        params![now, code],
    )?;
    if changed != 1 {
        anyhow::bail!("authorization code already consumed");
    }
    Ok(ConsumedAuthCode {
        code: row.0,
        client_id: row.1,
        redirect_uri: row.2,
        code_challenge: row.3,
        expires_at: row.4,
    })
}

pub fn insert_grant(conn: &Connection, input: InsertGrantInput) -> anyhow::Result<GrantRecord> {
    let grant_id = ulid::Ulid::new().to_string();
    let refresh_token_hash = hash_secret(&input.refresh_token)?;
    let now = chrono::Utc::now().timestamp();
    let changed = conn.execute(
        "INSERT INTO oauth_grants (
            grant_id, client_id, access_token_jti, access_expires_at,
            refresh_token_hash, refresh_expires_at, issued_at, revoked_at
         )
         SELECT ?1, c.client_id, ?3, ?4, ?5, ?6, ?7, NULL
         FROM oauth_clients c
         WHERE c.client_id = ?2 AND c.revoked_at IS NULL",
        params![
            &grant_id,
            &input.client_id,
            &input.access_token_jti,
            input.access_expires_at,
            &refresh_token_hash,
            input.refresh_expires_at,
            now
        ],
    )?;
    if changed != 1 {
        anyhow::bail!("client is revoked or unknown");
    }
    Ok(GrantRecord {
        grant_id,
        client_id: input.client_id,
        access_token_jti: input.access_token_jti,
        access_expires_at: input.access_expires_at,
        refresh_expires_at: input.refresh_expires_at,
        revoked_at: None,
    })
}

pub fn find_active_grant_by_refresh(
    conn: &Connection,
    client_id: &str,
    refresh_token: &str,
    now: i64,
) -> anyhow::Result<Option<GrantRecord>> {
    Ok(
        match find_grant_by_refresh(conn, client_id, refresh_token, now)? {
            RefreshGrantLookup::Active(grant) => Some(grant),
            RefreshGrantLookup::ReusedOrExpired | RefreshGrantLookup::NotFound => None,
        },
    )
}

pub fn find_grant_by_refresh(
    conn: &Connection,
    client_id: &str,
    refresh_token: &str,
    now: i64,
) -> anyhow::Result<RefreshGrantLookup> {
    let mut stmt = conn.prepare(
        "SELECT g.grant_id, g.client_id, g.access_token_jti, g.access_expires_at,
                g.refresh_token_hash, g.refresh_expires_at, g.revoked_at
         FROM oauth_grants g
         JOIN oauth_clients c ON c.client_id = g.client_id AND c.revoked_at IS NULL
         WHERE g.client_id = ?1",
    )?;
    let mut rows = stmt.query(params![client_id])?;
    while let Some(row) = rows.next()? {
        let hash: String = row.get(4)?;
        if verify_secret(refresh_token, &hash) {
            let grant = GrantRecord {
                grant_id: row.get(0)?,
                client_id: row.get(1)?,
                access_token_jti: row.get(2)?,
                access_expires_at: row.get(3)?,
                refresh_expires_at: row.get(5)?,
                revoked_at: row.get(6)?,
            };
            if grant.revoked_at.is_none() && grant.refresh_expires_at > now {
                return Ok(RefreshGrantLookup::Active(grant));
            }
            return Ok(RefreshGrantLookup::ReusedOrExpired);
        }
    }
    Ok(RefreshGrantLookup::NotFound)
}

pub fn consume_refresh_grant_for_rotation(
    conn: &Connection,
    client_id: &str,
    refresh_token: &str,
    now: i64,
) -> anyhow::Result<RefreshGrantLookup> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let lookup = find_grant_by_refresh(conn, client_id, refresh_token, now);
    match lookup {
        Ok(RefreshGrantLookup::Active(grant)) => {
            let changed = conn.execute(
                "UPDATE oauth_grants SET revoked_at = ?1
                 WHERE grant_id = ?2 AND revoked_at IS NULL",
                params![now, grant.grant_id],
            );
            match changed {
                Ok(1) => {
                    conn.execute_batch("COMMIT")?;
                    Ok(RefreshGrantLookup::Active(grant))
                }
                Ok(_) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Ok(RefreshGrantLookup::ReusedOrExpired)
                }
                Err(err) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Err(err.into())
                }
            }
        }
        Ok(other) => {
            conn.execute_batch("COMMIT")?;
            Ok(other)
        }
        Err(err) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(err)
        }
    }
}

pub fn rotate_refresh_grant(
    conn: &Connection,
    client_id: &str,
    refresh_token: &str,
    new_grant: InsertGrantInput,
    now: i64,
) -> anyhow::Result<RefreshGrantLookup> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let lookup = find_grant_by_refresh(conn, client_id, refresh_token, now);
    match lookup {
        Ok(RefreshGrantLookup::Active(old_grant)) => {
            let changed = conn.execute(
                "UPDATE oauth_grants SET revoked_at = ?1
                 WHERE grant_id = ?2 AND revoked_at IS NULL",
                params![now, old_grant.grant_id],
            );
            match changed {
                Ok(1) => match insert_grant(conn, new_grant) {
                    Ok(inserted) => {
                        conn.execute_batch("COMMIT")?;
                        Ok(RefreshGrantLookup::Active(inserted))
                    }
                    Err(err) => {
                        let _ = conn.execute_batch("ROLLBACK");
                        Err(err)
                    }
                },
                Ok(_) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Ok(RefreshGrantLookup::ReusedOrExpired)
                }
                Err(err) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Err(err.into())
                }
            }
        }
        Ok(other) => {
            conn.execute_batch("COMMIT")?;
            Ok(other)
        }
        Err(err) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(err)
        }
    }
}

pub fn active_grant_by_jti(
    conn: &Connection,
    jti: &str,
    now: i64,
) -> anyhow::Result<Option<GrantRecord>> {
    conn.query_row(
        "SELECT g.grant_id, g.client_id, g.access_token_jti, g.access_expires_at,
                g.refresh_expires_at, g.revoked_at
         FROM oauth_grants g
         JOIN oauth_clients c ON c.client_id = g.client_id AND c.revoked_at IS NULL
         WHERE g.access_token_jti = ?1 AND g.revoked_at IS NULL AND g.access_expires_at > ?2",
        params![jti, now],
        |row| {
            Ok(GrantRecord {
                grant_id: row.get(0)?,
                client_id: row.get(1)?,
                access_token_jti: row.get(2)?,
                access_expires_at: row.get(3)?,
                refresh_expires_at: row.get(4)?,
                revoked_at: row.get(5)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn revoke_grant(conn: &Connection, grant_id: &str) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE oauth_grants SET revoked_at = COALESCE(revoked_at, ?1)
         WHERE grant_id = ?2",
        params![chrono::Utc::now().timestamp(), grant_id],
    )?;
    Ok(())
}

pub fn revoke_grants_for_client(conn: &Connection, client_id: &str) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE oauth_grants SET revoked_at = COALESCE(revoked_at, ?1)
         WHERE client_id = ?2",
        params![chrono::Utc::now().timestamp(), client_id],
    )?;
    Ok(())
}

pub fn active_grant_count_for_client(conn: &Connection, client_id: &str) -> anyhow::Result<i64> {
    let now = chrono::Utc::now().timestamp();
    conn.query_row(
        "SELECT COUNT(*) FROM oauth_grants
         WHERE client_id = ?1 AND revoked_at IS NULL AND refresh_expires_at > ?2",
        params![client_id, now],
        |row| row.get(0),
    )
    .map_err(Into::into)
}

pub fn current_signing_key(conn: &Connection) -> anyhow::Result<SigningKey> {
    conn.query_row(
        "SELECT kid, secret_hex, created_at, rotated_at
         FROM oauth_signing_keys
         ORDER BY rotated_at IS NULL DESC, created_at DESC
         LIMIT 1",
        [],
        |row| {
            Ok(SigningKey {
                kid: row.get(0)?,
                secret_hex: row.get(1)?,
                created_at: row.get(2)?,
                rotated_at: row.get(3)?,
            })
        },
    )
    .map_err(Into::into)
}

pub fn signing_keys_for_verification(conn: &Connection) -> anyhow::Result<Vec<SigningKey>> {
    let cutoff = chrono::Utc::now().timestamp() - 3600;
    let mut stmt = conn.prepare(
        "SELECT kid, secret_hex, created_at, rotated_at
         FROM oauth_signing_keys
         WHERE rotated_at IS NULL OR rotated_at >= ?1
         ORDER BY created_at DESC",
    )?;
    let rows = stmt.query_map([cutoff], |row| {
        Ok(SigningKey {
            kid: row.get(0)?,
            secret_hex: row.get(1)?,
            created_at: row.get(2)?,
            rotated_at: row.get(3)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn mark_client_used(conn: &Connection, client_id: &str) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE oauth_clients SET last_used_at = ?1 WHERE client_id = ?2",
        params![chrono::Utc::now().timestamp(), client_id],
    )?;
    Ok(())
}

pub fn revoke_grant_by_access_jti(conn: &Connection, jti: &str) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE oauth_grants SET revoked_at = COALESCE(revoked_at, ?1)
         WHERE access_token_jti = ?2",
        params![chrono::Utc::now().timestamp(), jti],
    )?;
    Ok(())
}

pub fn revoke_client(conn: &Connection, client_id: &str) -> anyhow::Result<()> {
    let now = chrono::Utc::now().timestamp();
    conn.execute(
        "UPDATE oauth_clients SET revoked_at = COALESCE(revoked_at, ?1)
         WHERE client_id = ?2",
        params![now, client_id],
    )?;
    conn.execute(
        "UPDATE oauth_grants SET revoked_at = COALESCE(revoked_at, ?1)
         WHERE client_id = ?2",
        params![now, client_id],
    )?;
    Ok(())
}

pub fn list_clients(conn: &Connection) -> anyhow::Result<Vec<ClientSummary>> {
    let mut stmt = conn.prepare(
        "SELECT c.client_id, c.client_name, c.registered_at, c.last_used_at, c.revoked_at,
                (SELECT COUNT(*) FROM oauth_grants g
                 WHERE g.client_id = c.client_id
                   AND g.revoked_at IS NULL
                   AND g.refresh_expires_at > ?1) AS active_grants
         FROM oauth_clients c
         ORDER BY c.registered_at DESC",
    )?;
    let now = chrono::Utc::now().timestamp();
    let rows = stmt.query_map([now], |row| {
        Ok(ClientSummary {
            client_id: row.get(0)?,
            client_name: row.get(1)?,
            registered_at: row.get(2)?,
            last_used_at: row.get(3)?,
            revoked_at: row.get(4)?,
            active_grants: row.get(5)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn gc_expired_oauth_records(conn: &Connection, now: i64) -> anyhow::Result<(usize, usize)> {
    let old_codes = conn.execute(
        "DELETE FROM oauth_auth_codes WHERE expires_at < ?1",
        [now - 86_400],
    )?;
    let old_grants = conn.execute(
        "DELETE FROM oauth_grants
         WHERE access_expires_at < ?1 AND refresh_expires_at < ?2",
        [now - 86_400 * 30, now - 86_400 * 30],
    )?;
    Ok((old_codes, old_grants))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> crate::store::sqlite::SqliteStore {
        crate::store::sqlite::SqliteStore::in_memory().expect("in-memory store")
    }

    #[test]
    fn register_client_hashes_secret_and_allows_duplicate_name() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/api/mcp/auth_callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "client_secret_basic".to_string(),
            },
        )
        .expect("register client");

        assert_eq!(registered.client_name, "claude.ai");
        assert!(registered
            .client_secret
            .as_deref()
            .is_some_and(|s| s.len() >= 32));

        let raw_secret_rows: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM oauth_clients WHERE client_secret_hash = ?1",
                [registered.client_secret.as_deref().unwrap()],
                |row| row.get(0),
            )
            .expect("query secret");
        assert_eq!(raw_secret_rows, 0, "secret must not be stored in plaintext");

        let duplicate = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/api/mcp/auth_callback".to_string()],
                grant_types: vec!["authorization_code".to_string()],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("duplicate display names are allowed");
        assert_ne!(duplicate.client_id, registered.client_id);
        assert_eq!(duplicate.client_name, "claude.ai");
    }

    #[test]
    fn register_client_accepts_ipv6_loopback_redirect_uri() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "local-callback".to_string(),
                redirect_uris: vec!["http://[::1]:1234/callback".to_string()],
                grant_types: vec!["authorization_code".to_string()],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("IPv6 loopback callback should be accepted");

        assert_eq!(
            registered.redirect_uris,
            vec!["http://[::1]:1234/callback".to_string()]
        );
    }

    #[test]
    fn register_client_normalizes_duplicate_grant_types() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "duplicate-grants".to_string(),
                redirect_uris: vec!["https://client.example/callback".to_string()],
                grant_types: vec![
                    "refresh_token".to_string(),
                    "authorization_code".to_string(),
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("duplicate grant metadata should be normalized");

        assert_eq!(
            registered.grant_types,
            vec![
                "authorization_code".to_string(),
                "refresh_token".to_string()
            ]
        );

        let stored = get_client(store.conn(), &registered.client_id)
            .expect("load stored client")
            .expect("stored client");
        assert_eq!(stored.grant_types, registered.grant_types);
    }

    #[test]
    fn register_client_rejects_redirect_uri_fragment() {
        let store = test_store();
        let err = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "fragment-callback".to_string(),
                redirect_uris: vec!["https://client.example/callback#fragment".to_string()],
                grant_types: vec!["authorization_code".to_string()],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect_err("redirect fragments must be rejected");

        assert!(err.to_string().contains("fragment"));
    }

    #[test]
    fn register_client_rejects_oversized_public_metadata() {
        let store = test_store();
        let long_name = "a".repeat(129);
        let err = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: long_name,
                redirect_uris: vec!["https://client.example/callback".to_string()],
                grant_types: vec!["authorization_code".to_string()],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect_err("oversized client_name must be rejected");
        assert!(err.to_string().contains("client_name"));

        let err = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "too-many-redirects".to_string(),
                redirect_uris: (0..11)
                    .map(|idx| format!("https://client.example/callback/{idx}"))
                    .collect(),
                grant_types: vec!["authorization_code".to_string()],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect_err("too many redirect_uris must be rejected");
        assert!(err.to_string().contains("redirect_uris"));

        let long_uri = format!("https://client.example/{}", "x".repeat(2049));
        let err = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "long-redirect".to_string(),
                redirect_uris: vec![long_uri],
                grant_types: vec!["authorization_code".to_string()],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect_err("oversized redirect_uri must be rejected");
        assert!(err.to_string().contains("redirect_uri"));
    }

    #[test]
    fn register_client_prunes_unused_clients_to_durable_cap() {
        let store = test_store();
        for idx in 0..300 {
            register_client(
                store.conn(),
                RegisterClientInput {
                    client_name: format!("unused-client-{idx}"),
                    redirect_uris: vec![format!("https://client.example/callback/{idx}")],
                    grant_types: vec!["authorization_code".to_string()],
                    token_endpoint_auth_method: "none".to_string(),
                },
            )
            .expect("register unused client");
        }

        let client_count: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM oauth_clients", [], |row| row.get(0))
            .expect("count clients");
        assert!(
            client_count <= 256,
            "unused public DCR clients must be durably capped, got {client_count}"
        );
    }

    #[test]
    fn register_client_prunes_revoked_clients_to_durable_cap() {
        let store = test_store();
        for idx in 0..300 {
            let registered = register_client(
                store.conn(),
                RegisterClientInput {
                    client_name: format!("revoked-client-{idx}"),
                    redirect_uris: vec![format!("https://client.example/callback/{idx}")],
                    grant_types: vec!["authorization_code".to_string()],
                    token_endpoint_auth_method: "none".to_string(),
                },
            )
            .expect("register client");
            revoke_client(store.conn(), &registered.client_id).expect("revoke client");
        }

        let client_count: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM oauth_clients", [], |row| row.get(0))
            .expect("count clients");
        assert!(
            client_count <= 256,
            "revoked public DCR clients must be durably capped, got {client_count}"
        );
    }

    #[test]
    fn auth_code_is_single_use_and_exact_redirect_bound() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec!["authorization_code".to_string()],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("register client");
        let code = insert_auth_code(
            store.conn(),
            InsertAuthCodeInput {
                client_id: registered.client_id.clone(),
                redirect_uri: "https://claude.ai/callback".to_string(),
                code_challenge: "challenge".to_string(),
                expires_at: chrono::Utc::now().timestamp() + 600,
            },
        )
        .expect("insert code");

        assert!(consume_auth_code(
            store.conn(),
            &code,
            &registered.client_id,
            "https://claude.ai/other",
            chrono::Utc::now().timestamp(),
        )
        .is_err());

        let consumed = consume_auth_code(
            store.conn(),
            &code,
            &registered.client_id,
            "https://claude.ai/callback",
            chrono::Utc::now().timestamp(),
        )
        .expect("consume code");
        assert_eq!(consumed.code_challenge, "challenge");

        assert!(consume_auth_code(
            store.conn(),
            &code,
            &registered.client_id,
            "https://claude.ai/callback",
            chrono::Utc::now().timestamp(),
        )
        .is_err());
    }

    #[test]
    fn refresh_rotation_revokes_old_grant_and_replay_revokes_client_family() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("register client");
        let refresh = "refresh-one";
        let grant = insert_grant(
            store.conn(),
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "jti-one".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: refresh.to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86400,
            },
        )
        .expect("insert grant");

        let found = find_active_grant_by_refresh(
            store.conn(),
            &registered.client_id,
            refresh,
            chrono::Utc::now().timestamp(),
        )
        .expect("find grant")
        .expect("grant should exist");
        assert_eq!(found.grant_id, grant.grant_id);

        revoke_grant(store.conn(), &grant.grant_id).expect("revoke old grant");
        let replay = find_active_grant_by_refresh(
            store.conn(),
            &registered.client_id,
            refresh,
            chrono::Utc::now().timestamp(),
        )
        .expect("lookup replay");
        assert!(
            replay.is_none(),
            "revoked refresh token should not be active"
        );

        revoke_grants_for_client(store.conn(), &registered.client_id).expect("revoke family");
        let active = active_grant_count_for_client(store.conn(), &registered.client_id)
            .expect("count active grants");
        assert_eq!(active, 0);
    }

    #[test]
    fn grants_cannot_be_inserted_or_verified_for_revoked_clients() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("register client");
        insert_grant(
            store.conn(),
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "jti-before-revoke".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "refresh-before-revoke".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86_400,
            },
        )
        .expect("insert grant before revoke");
        store
            .conn()
            .execute(
                "UPDATE oauth_clients SET revoked_at = ?1 WHERE client_id = ?2",
                rusqlite::params![chrono::Utc::now().timestamp(), registered.client_id],
            )
            .expect("mark client revoked without sweeping grants");

        let active = active_grant_by_jti(
            store.conn(),
            "jti-before-revoke",
            chrono::Utc::now().timestamp(),
        )
        .expect("lookup grant by jti");
        assert!(
            active.is_none(),
            "bearer verification must reject grants owned by revoked clients"
        );

        let insert_after_revoke = insert_grant(
            store.conn(),
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "jti-after-revoke".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "refresh-after-revoke".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86_400,
            },
        );
        assert!(
            insert_after_revoke.is_err(),
            "token exchange must not insert a new grant after client revocation"
        );
    }

    #[test]
    fn list_clients_excludes_expired_grants_from_active_count() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("register client");
        let now = chrono::Utc::now().timestamp();
        insert_grant(
            store.conn(),
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "expired-jti".to_string(),
                access_expires_at: now - 120,
                refresh_token: "expired-refresh".to_string(),
                refresh_expires_at: now - 60,
            },
        )
        .expect("insert expired grant");
        insert_grant(
            store.conn(),
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "active-jti".to_string(),
                access_expires_at: now + 3600,
                refresh_token: "active-refresh".to_string(),
                refresh_expires_at: now + 86_400,
            },
        )
        .expect("insert active grant");

        let clients = list_clients(store.conn()).expect("list clients");

        assert_eq!(clients.len(), 1);
        assert_eq!(clients[0].active_grants, 1);
    }

    #[test]
    fn consume_refresh_grant_is_single_use() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("register client");
        let refresh = "refresh-one";
        let grant = insert_grant(
            store.conn(),
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "jti-consume".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: refresh.to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86400,
            },
        )
        .expect("insert grant");

        let consumed = consume_refresh_grant_for_rotation(
            store.conn(),
            &registered.client_id,
            refresh,
            chrono::Utc::now().timestamp(),
        )
        .expect("consume refresh");
        assert!(
            matches!(consumed, RefreshGrantLookup::Active(active) if active.grant_id == grant.grant_id)
        );

        let replay = consume_refresh_grant_for_rotation(
            store.conn(),
            &registered.client_id,
            refresh,
            chrono::Utc::now().timestamp(),
        )
        .expect("lookup replay");
        assert!(matches!(replay, RefreshGrantLookup::ReusedOrExpired));
    }

    #[test]
    fn rotate_refresh_grant_inserts_replacement_atomically() {
        let store = test_store();
        let registered = register_client(
            store.conn(),
            RegisterClientInput {
                client_name: "claude.ai".to_string(),
                redirect_uris: vec!["https://claude.ai/callback".to_string()],
                grant_types: vec![
                    "authorization_code".to_string(),
                    "refresh_token".to_string(),
                ],
                token_endpoint_auth_method: "none".to_string(),
            },
        )
        .expect("register client");
        let refresh = "refresh-one";
        insert_grant(
            store.conn(),
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "jti-before".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: refresh.to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86400,
            },
        )
        .expect("insert grant");

        let rotated = rotate_refresh_grant(
            store.conn(),
            &registered.client_id,
            refresh,
            InsertGrantInput {
                client_id: registered.client_id.clone(),
                access_token_jti: "jti-after".to_string(),
                access_expires_at: chrono::Utc::now().timestamp() + 3600,
                refresh_token: "refresh-two".to_string(),
                refresh_expires_at: chrono::Utc::now().timestamp() + 86400,
            },
            chrono::Utc::now().timestamp(),
        )
        .expect("rotate refresh");

        assert!(
            matches!(rotated, RefreshGrantLookup::Active(active) if active.access_token_jti == "jti-after")
        );
        let active = active_grant_count_for_client(store.conn(), &registered.client_id)
            .expect("count active grants");
        assert_eq!(active, 1);
    }
}
