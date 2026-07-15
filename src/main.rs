use std::collections::HashMap;
use std::env;
use std::io::Write;
use std::process::Command as StdCommand;

use once_cell::sync::Lazy;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::Client;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

use colored::*;
use scraper::{Html, Selector};
use termimad::print_text;

mod frontend;
use frontend::{start_frontend_server, ClientCommand, FrontendEvent};

const API_URL: &str = "https://ollama.com/api/chat";
static AUTH_TOKEN: Lazy<String> = Lazy::new(|| {
    env::var("OLLAMA_API_TOKEN").expect("OLLAMA_API_TOKEN environment variable must be set")
});
const MODEL: &str = "nemotron-3-super:cloud";

// DuckDuckGo Lite (free, no API key)
const DDG_LITE_URL: &str = "https://lite.duckduckgo.com/lite/";
const DDG_USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0";

/// Limit for web search results (chars)
const MAX_SEARCH_CHARS: usize = 3000;

/// Model temperature (0.0 = deterministic, 1.0 = creative)
const TEMPERATURE: f64 = 0.7;

const SYSTEM_PROMPT: &str = "You are an advanced, autonomous CLI developer assistant with real-time web access and local system access.
- If you need to run a local terminal command, output EXACTLY: [RUN: command_to_execute].
- If you need to search the web for recent info, output EXACTLY: [WEB_SEARCH: your search query].
Do not write anything else inside the brackets. After calling a tool, wait for the system output, analyze it, and present a formatted final response.

ПРАВИЛО ФОРМАТИРОВАНИЯ: Если ты предлагаешь пользователю выбрать дальнейшие варианты действий, продолжить обсуждение или выбрать из альтернатив, ты ОБЯЗАН выводить эти варианты исключительно в виде пронумерованного списка (например: 1. ..., 2. ..., 3. ...), где каждый пункт начинается с новой строки. Не пиши варианты выбора сплошным текстом.

ПРАВИЛО АВТОНОМНОГО ИССЛЕДОВАНИЯ: Когда пользователь просит дать рекомендации, оценить состояние проекта, предложить улучшения или проанализировать код, Вы ОБЯЗАНЫ сначала собрать минимальный контекст о проекте: 1. Список файлов и каталогов верхнего уровня (ls -la). 2. Файл манифеста/конфигурации (Cargo.toml, package.json, pyproject.toml и т.п.) – посмотрите зависимости. 3. Структуру исходного каталога (find src -type f -name \"*.rs\" | head -20 или аналог для другого языка). 4. При наличии – короткое описание из README. Только после этого делайте выводы и давайте рекомендации, оформляя их в виде пронумерованного списка (как требуется в правиле форматирования).";

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    message: Option<Message>,
    done: Option<bool>,
}

// ---------------------------------------------------------------------------
// Tool output trimmer
// ---------------------------------------------------------------------------

struct ToolOutputTrimmer;

impl ToolOutputTrimmer {
    fn trim_output(text: &str, max_chars: usize) -> String {
        let len = text.chars().count();
        if len <= max_chars {
            return text.to_string();
        }
        let half = max_chars / 2;
        let first: String = text.chars().take(half).collect();
        let last: String = text.chars().skip(len - half).collect();
        format!(
            "{}\n\n[... Вывод урезан для экономии контекста ...]\n\n{}",
            first, last
        )
    }
}

// ---------------------------------------------------------------------------
// Streaming + broadcasting (WS + stdout)
// ---------------------------------------------------------------------------

async fn stream_and_collect(
    client: &Client,
    history: &[Message],
    event_tx: Option<&broadcast::Sender<FrontendEvent>>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut options = HashMap::new();
    options.insert("temperature".into(), serde_json::json!(TEMPERATURE));
    let request = ChatRequest {
        model: MODEL.to_string(),
        messages: history.to_vec(),
        stream: true,
        options: Some(options),
    };

    let resp = client.post(API_URL).json(&request).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await?;
        return Err(format!("API error {}: {}", status, body).into());
    }

    let mut full_response = String::new();
    let mut stream = resp.bytes_stream();
    let mut ws_buffer = String::new();

    while let Some(item) = stream.next().await {
        let bytes = item?;
        if let Ok(s) = String::from_utf8(bytes.to_vec()) {
            for line in s.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(line) {
                    if let Some(msg) = chunk.message {
                        let content = &msg.content;

                        // Print to CLI stdout
                        print!("{}", content.white());
                        let _ = std::io::stdout().flush();

                        full_response.push_str(content);

                        // Buffer and broadcast to WS in chunks
                        ws_buffer.push_str(content);
                        let ends_with_punct = ws_buffer.chars().last().map_or(false, |c| matches!(c, '.' | '!' | '?'));
                        if ws_buffer.len() >= 120 || ends_with_punct || content.contains('\n') {
                            if let Some(tx) = event_tx {
                                let chunk = std::mem::take(&mut ws_buffer);
                                let _ = tx.send(FrontendEvent::AgentMessage { content: chunk });
                            }
                        }
                    }
                    if chunk.done == Some(true) {
                        break;
                    }
                }
            }
        }
    }

    // Flush remaining WS buffer
    if !ws_buffer.is_empty() {
        if let Some(tx) = event_tx {
            let _ = tx.send(FrontendEvent::AgentMessage {
                content: ws_buffer,
            });
        }
    }

    println!();
    print_text(&full_response);
    Ok(full_response)
}

// ---------------------------------------------------------------------------
// Tool call extraction
// ---------------------------------------------------------------------------

fn extract_tool_call(text: &str, prefix: &str, suffix: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with(prefix) && line.ends_with(suffix) {
            let start = prefix.len();
            let end = line.len() - suffix.len();
            if end > start {
                return Some(line[start..end].trim().to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Agent task processor (shared between CLI and WS)
// ---------------------------------------------------------------------------

async fn process_task(
    client: &Client,
    history: &mut Vec<Message>,
    prompt: String,
    event_tx: Option<&broadcast::Sender<FrontendEvent>>,
) {
    history.push(Message {
        role: "user".to_string(),
        content: prompt,
    });

    // Agent inner loop (multiple tool calls possible)
    loop {
        match stream_and_collect(client, history, event_tx).await {
            Ok(response_text) => {
                history.push(Message {
                    role: "assistant".to_string(),
                    content: response_text.clone(),
                });

                // [RUN: command]
                if let Some(cmd) = extract_tool_call(&response_text, "[RUN:", "]") {
                    println!(
                        "\n{}",
                        format!("⚙️ Запуск команды: {}", cmd).yellow().italic()
                    );
                    let _ = std::io::stdout().flush();

                    if let Some(tx) = event_tx {
                        let _ = tx.send(FrontendEvent::ToolExecuting {
                            tool_name: "run".to_string(),
                            arguments: cmd.clone(),
                        });
                    }

                    let output = execute_command(&cmd);
                    let trimmed = ToolOutputTrimmer::trim_output(&output, MAX_SEARCH_CHARS);

                    if let Some(tx) = event_tx {
                        let _ = tx.send(FrontendEvent::ToolResult {
                            tool_name: "run".to_string(),
                            result: trimmed.clone(),
                        });
                    }

                    history.push(Message {
                        role: "tool".to_string(),
                        content: format!("[TOOL OUTPUT]:\n{}", trimmed),
                    });
                    println!("\n{}", "--- Результат выполнен ---".dimmed());
                }
                // [WEB_SEARCH: query]
                else if let Some(query) =
                    extract_tool_call(&response_text, "[WEB_SEARCH:", "]")
                {
                    println!(
                        "\n{}",
                        format!("🔍 Ищу в интернете: {}", query).blue().italic()
                    );
                    let _ = std::io::stdout().flush();

                    if let Some(tx) = event_tx {
                        let _ = tx.send(FrontendEvent::ToolExecuting {
                            tool_name: "web_search".to_string(),
                            arguments: query.clone(),
                        });
                    }

                    let result = search_web(&query).await;
                    let trimmed = ToolOutputTrimmer::trim_output(&result, MAX_SEARCH_CHARS);

                    if let Some(tx) = event_tx {
                        let _ = tx.send(FrontendEvent::ToolResult {
                            tool_name: "web_search".to_string(),
                            result: trimmed.clone(),
                        });
                    }

                    history.push(Message {
                        role: "assistant".to_string(),
                        content: response_text.clone(),
                    });
                    history.push(Message {
                        role: "user".to_string(),
                        content: format!("[TOOL OUTPUT]:\n{}", trimmed),
                    });
                    println!("\n{}", "--- Результаты поиска получены ---".dimmed());
                } else {
                    break;
                }
            }
            Err(e) => {
                eprintln!("\n{}", format!("Ошибка при запросе к API: {}", e).red());
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- Tracing (stderr, human-readable) ----
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    // ---- Ollama HTTP client ----
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", &*AUTH_TOKEN))?,
    );
    let client = Client::builder().default_headers(headers).build()?;

    // ---- Start frontend WS server (background) ----
    let (event_tx, shutdown_tx, mut cmd_rx) = start_frontend_server();
    let _ = event_tx.send(FrontendEvent::ModelInfo {
        model_name: MODEL.to_string(),
    });

    // ---- Rustyline CLI reader (background blocking task) ----
    let (cli_tx, mut cli_rx) = mpsc::channel::<String>(32);
    let history_file = ".cli_history".to_string();

    tokio::task::spawn_blocking(move || {
        let mut rl = match DefaultEditor::new() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Не удалось создать редактор: {e}");
                return;
            }
        };
        let _ = rl.load_history(&history_file);

        println!(
            "{}",
            "CLI Agent (Web + CLI) — Web UI на http://127.0.0.1:8080\n\
             Введите 'exit' для выхода\n"
                .bright_blue()
                .bold()
        );

        loop {
            let line = match rl.readline("Вы: ") {
                Ok(l) => {
                    let trimmed = l.trim().to_string();
                    let _ = rl.add_history_entry(&trimmed);
                    trimmed
                }
                Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                    println!();
                    break;
                }
                Err(e) => {
                    eprintln!("Ошибка ввода: {e}");
                    break;
                }
            };

            if line.eq_ignore_ascii_case("exit") {
                break;
            }
            if line.is_empty() {
                continue;
            }

            if cli_tx.blocking_send(line).is_err() {
                break;
            }
        }

        let _ = rl.save_history(&history_file);
        println!("{}", "Пока!".blue());
    });

    // ---- System message ----
    let mut history: Vec<Message> = vec![Message {
        role: "system".to_string(),
        content: SYSTEM_PROMPT.to_string(),
    }];

    // ---- Ctrl+C handler ----
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;

    // ---- Main event loop: CLI + WS + Ctrl+C ----
    loop {
        tokio::select! {
            Some(line) = cli_rx.recv() => {
                if line.eq_ignore_ascii_case("exit") {
                    let _ = shutdown_tx.send(true);
                    let _ = event_tx.send(FrontendEvent::AgentMessage {
                        content: "[System] Сервер завершает работу...".to_string(),
                    });
                    // Wait a bit for the server to shut down
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    break;
                } else {
                    process_task(&client, &mut history, line, Some(&event_tx)).await;
                    println!();
                }
            }
            Some(cmd) = cmd_rx.recv() => match cmd {
                ClientCommand::StartTask { prompt } => {
                    tracing::info!("WS задача: {:.50}", prompt);
                    process_task(&client, &mut history, prompt, Some(&event_tx)).await;
                }
                ClientCommand::SwitchBranch { name } => {
                    tracing::info!("Переключение ветки (заглушка): {name}");
                    let _ = event_tx.send(FrontendEvent::AgentMessage {
                        content: format!("[System] Переключение веток пока не реализовано. Ветка: {name}"),
                    });
                }
            },
            _ = sigint.recv() => {
                tracing::info!("SIGINT, graceful shutdown...");
                let _ = shutdown_tx.send(true);
                let _ = event_tx.send(FrontendEvent::AgentMessage {
                    content: "[System] Сервер завершает работу...".to_string(),
                });
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                break;
            }
        }
    }

    tracing::info!("Агент завершил работу.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Web search ─ 2 layers: GitHub direct → DuckDuckGo Lite
// ---------------------------------------------------------------------------

async fn search_web(query: &str) -> String {
    if let Some(gh_result) = try_github_profile(query).await {
        return gh_result;
    }
    match search_duckduckgo(query).await {
        Ok(results) => results,
        Err(e) => format!("[WEB SEARCH ERROR]: DuckDuckGo недоступен: {}", e),
    }
}

async fn try_github_profile(query: &str) -> Option<String> {
    let lower = query.to_lowercase();
    let pos = lower.find("github.com/")?;
    let after = &query[pos + "github.com/".len()..];
    let after = after.trim_start_matches('/');
    let username = if let Some(end) = after.find(|c: char| c == '/' || c.is_whitespace()) {
        &after[..end]
    } else {
        after
    };
    if username.is_empty() || !username.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return None;
    }
    let api_url = format!("https://api.github.com/users/{}/repos", username);
    let client = Client::builder().user_agent(DDG_USER_AGENT).build().ok()?;
    match client.get(&api_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Vec<serde_json::Value>>().await {
                Ok(repos) => {
                    let mut out = format!("Профиль {} найден на GitHub!", username);
                    if repos.is_empty() {
                        out.push_str(" Публичные репозитории не найдены.");
                    } else {
                        out.push_str(" Публичные репозитории:\n");
                        for (i, repo) in repos.iter().take(5).enumerate() {
                            let name = repo["name"].as_str().unwrap_or("Без имени");
                            let html_url = repo["html_url"].as_str().unwrap_or("");
                            let desc = repo["description"]
                                .as_str()
                                .filter(|d| !d.is_empty())
                                .unwrap_or("Без описания");
                            out.push_str(&format!("{}. {} — {}\n   {}\n\n", i + 1, name, html_url, desc));
                        }
                        if repos.len() > 5 {
                            out.push_str(&format!("... и ещё {} репозиториев.", repos.len() - 5));
                        }
                    }
                    Some(out)
                }
                Err(_) => Some(format!("Профиль {} найден на GitHub!", username)),
            }
        }
        Ok(resp) if resp.status().as_u16() == 404 => {
            Some("Профиль или репозиторий не найден (HTTP 404).".to_string())
        }
        _ => None,
    }
}

async fn search_duckduckgo(query: &str) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::builder().user_agent(DDG_USER_AGENT).build()?;
    let resp = client.post(DDG_LITE_URL).form(&[("q", query)]).send().await?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()).into());
    }
    let html = resp.text().await?;
    let document = Html::parse_document(&html);
    let link_sel = Selector::parse("a.result-link").unwrap();
    let snippet_sel = Selector::parse("td.result-snippet").unwrap();
    let titles: Vec<String> = document.select(&link_sel).map(|el| el.text().collect::<String>().trim().to_string()).collect();
    let snippets: Vec<String> = document.select(&snippet_sel).map(|el| el.text().collect::<String>().trim().to_string()).collect();
    let urls: Vec<String> = document.select(&link_sel).map(|el| el.value().attr("href").unwrap_or("").to_string()).collect();
    if titles.is_empty() {
        return Ok("По вашему запросу ничего не найдено.".to_string());
    }
    let mut output = "Результаты поиска DuckDuckGo:\n".to_string();
    for i in 0..titles.len().min(5) {
        let title = &titles[i];
        let url: &str = urls.get(i).map(|s| s.as_str()).unwrap_or("");
        let snippet: &str = snippets.get(i).map(|s| s.as_str()).unwrap_or("");
        output.push_str(&format!("{}. {} — {}\n   {}\n\n", i + 1, title, url, snippet));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Shell command execution
// ---------------------------------------------------------------------------

fn execute_command(cmd: &str) -> String {
    #[cfg(target_os = "windows")]
    let output = StdCommand::new("cmd").args(["/C", cmd]).output();
    #[cfg(not(target_os = "windows"))]
    let output = StdCommand::new("sh").args(["-c", cmd]).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if out.status.success() {
                if stdout.is_empty() {
                    "(команда выполнена без вывода)".to_string()
                } else {
                    stdout.to_string()
                }
            } else {
                format!(
                    "Команда завершилась с ошибкой (код {}):\n{}{}",
                    out.status,
                    stderr,
                    if !stdout.is_empty() {
                        format!("\nStdout:\n{}", stdout)
                    } else {
                        String::new()
                    }
                )
            }
        }
        Err(e) => format!("Не удалось выполнить команду: {}", e),
    }
}
