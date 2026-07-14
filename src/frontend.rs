// ---------------------------------------------------------------------------
// Frontend — WebSocket-сервер для трансляции событий агента на UI
//
//   FrontendEvent      → типы событий (AgentMessage, ToolExecuting, …)
//   ClientCommand      → команды от UI к агенту (StartTask, …)
//   start_frontend_server() → запуск axum на 0.0.0.0:8080
// ---------------------------------------------------------------------------

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, watch};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use axum::routing::get_service;

// ---------------------------------------------------------------------------
// FrontendEvent
// ---------------------------------------------------------------------------

/// События, транслируемые на фронтенд через WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FrontendEvent {
    /// Текстовый ответ агента (промежуточный стриминг или финальный).
    AgentMessage {
        content: String,
    },
    /// Heartbeat — держит соединение живым.
    Ping,
    /// Агент начал выполнение инструмента.
    ToolExecuting {
        tool_name: String,
        arguments: String,
    },
    /// Результат выполнения инструмента.
    ToolResult {
        tool_name: String,
        result: String,
    },
    /// Информация о модели при запуске.
    ModelInfo {
        model_name: String,
    },
}

// ---------------------------------------------------------------------------
// ClientCommand — команды от фронтенда к агенту
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientCommand {
    /// Запустить новую задачу.
    StartTask {
        prompt: String,
    },
    /// Переключиться на ветку контекста (заглушка).
    SwitchBranch {
        name: String,
    },
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<FrontendEvent>,
    cmd_tx: mpsc::Sender<ClientCommand>,
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Обслуживает одно WebSocket-соединение.
/// broadcast → клиент: все FrontendEvent отправляются как JSON.
/// клиент → mpsc: команды отправляются в cmd_tx.
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();
    let cmd_tx = state.cmd_tx;

    // Heartbeat: держит WebSocket живым во время долгих LLM-инференсов.
    let heartbeat_tx = state.tx.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        // skip first immediate tick
        interval.tick().await;
        loop {
            interval.tick().await;
            if heartbeat_tx.send(FrontendEvent::Ping).is_err() {
                break;
            }
        }
    });

    loop {
        tokio::select! {
            // broadcast → клиент
            event = rx.recv() => {
                match event {
                    Ok(event) => {
                        let json = match serde_json::to_string(&event) {
                            Ok(j) => j,
                            Err(_) => continue,
                        };
                        if sender.send(WsMessage::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("Frontend WS lagged by {n} events");
                        continue;
                    }
                }
            }
            // клиент → mpsc
            msg = receiver.next() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        if let Ok(cmd) = serde_json::from_str::<ClientCommand>(&text) {
                            let _ = cmd_tx.send(cmd).await;
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) | None => break,
                    Some(Err(e)) => {
                        tracing::warn!("Frontend WS receive error: {e}");
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Запуск сервера
// ---------------------------------------------------------------------------

/// Пытается освободить порт, убивая процесс, который его слушает.
fn free_port_local(port: u16) -> bool {
    use std::process::Command;
    use std::time::Duration;

    fn find_pid(port: u16) -> Option<i32> {
        // fuser - самый быстрый и точный (есть в psmisc, почти везде)
        let out = Command::new("fuser")
            .arg(format!("{port}/tcp"))
            .output()
            .ok()?;
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            return stdout
                .split_whitespace()
                .last()
                .and_then(|s| s.parse::<i32>().ok());
        }
        // lsof (fallback)
        let out = Command::new("sh")
            .arg("-c")
            .arg(format!("lsof -ti :{port} 2>/dev/null"))
            .output()
            .ok()?;
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            return stdout.trim().split('\n').last().and_then(|s| s.parse::<i32>().ok());
        }
        None
    }

    fn is_alive(pid: i32) -> bool {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    let pid = match find_pid(port) {
        Some(pid) => pid,
        None => return true,
    };

    tracing::warn!("Port {port} занят PID {pid}, отправляю SIGTERM...");
    let _ = Command::new("kill").arg(pid.to_string()).status();
    std::thread::sleep(Duration::from_millis(500));

    if is_alive(pid) {
        tracing::warn!("PID {pid} ещё жив, отправляю SIGKILL");
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
        std::thread::sleep(Duration::from_millis(100));
    }

    !is_alive(pid)
}

/// Запускает WebSocket-сервер + статику на `0.0.0.0:8080`.
///
/// Возвращает:
/// - `broadcast::Sender<FrontendEvent>` — публикация событий
/// - `watch::Sender<bool>` — graceful shutdown
/// - `mpsc::Receiver<ClientCommand>` — команды (StartTask, SwitchBranch) от фронтенда
pub fn start_frontend_server() -> (
    broadcast::Sender<FrontendEvent>,
    watch::Sender<bool>,
    mpsc::Receiver<ClientCommand>,
) {
    let (tx, _rx) = broadcast::channel(256);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let (cmd_tx, cmd_rx) = mpsc::channel(32);

    let state = AppState {
        tx: tx.clone(),
        cmd_tx,
    };

    let static_dir = std::env::var("CLI_AGENT_STATIC_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            manifest_dir.join("static")
        });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(get_service(
            ServeDir::new(&static_dir).append_index_html_on_directories(true),
        ))
        .layer(CorsLayer::permissive())
        .with_state(state);

    tokio::spawn(async move {
        free_port_local(8080);

        let listener = match tokio::net::TcpListener::bind("0.0.0.0:8080").await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Не удалось привязать фронтенд-сервер: {e}");
                return;
            }
        };

        tracing::info!("Фронтенд-сервер на http://127.0.0.1:8080");

        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let mut rx = shutdown_rx;
                rx.changed().await.ok();
            })
            .await
        {
            tracing::error!("Фронтенд-сервер ошибка: {e}");
        }
    });

    (tx, shutdown_tx, cmd_rx)
}
