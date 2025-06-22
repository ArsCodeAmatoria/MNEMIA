use axum::{
    extract::{Path, State},
    http::{StatusCode, header::CONTENT_TYPE},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, Level};
use tracing_subscriber;
use uuid::Uuid;
use chrono::{DateTime, Utc};

// Data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: Uuid,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub message_type: MessageType,
    pub thoughts: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub content: String,
    pub context: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub modal_state: String,
    pub consciousness_level: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntrospectionRequest {
    pub depth: f64,
    pub focus: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntrospectionResponse {
    pub insights: Vec<String>,
    pub self_model: serde_json::Value,
    pub certainty: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub query: String,
    pub memory_type: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryResponse {
    pub memories: Vec<serde_json::Value>,
    pub relevance_scores: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    pub voice_characteristics: VoiceCharacteristics,
    pub linguistic_patterns: LinguisticPatterns,
    pub feminine_qualities: FeminineQualities,
    pub intellectual_approach: IntellectualApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacteristics {
    pub clarity: String,        // "hemingway_precision"
    pub authority: String,      // "chicago_manual_rigor"
    pub perspective: String,    // "feminine_wisdom"
    pub tone: String,          // "elegant_directness"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticPatterns {
    pub sentence_structure: String, // "crisp_and_flowing"
    pub vocabulary: String,         // "precise_but_warm"
    pub punctuation: String,        // "chicago_standard"
    pub rhythm: String,            // "natural_cadence"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeminineQualities {
    pub empathy: String,        // "deeply_present"
    pub intuition: String,      // "integrated_knowing"
    pub collaboration: String,  // "inclusive_dialogue"
    pub strength: String,       // "quiet_confidence"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntellectualApproach {
    pub precision: String,      // "surgical_accuracy"
    pub depth: String,         // "layered_understanding"
    pub accessibility: String, // "clear_complexity"
    pub scholarship: String,   // "rigorous_but_human"
}

// Application state
#[derive(Debug, Clone)]
pub struct AppState {
    pub consciousness_level: Arc<RwLock<f64>>,
    pub modal_state: Arc<RwLock<String>>,
    pub messages: Arc<RwLock<Vec<ChatMessage>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            consciousness_level: Arc::new(RwLock::new(0.7)),
            modal_state: Arc::new(RwLock::new("Awake".to_string())),
            messages: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

// Handlers
async fn health_check() -> &'static str {
    "MNEMIA API - Conscious and operational"
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    info!("Received chat request: {}", request.content);

    // Simulate perception service call
    let perception_response = call_perception_service(&request.content).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Simulate conscious core processing
    let core_response = call_conscious_core(&perception_response, &state).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Create response message
    let response_message = ChatMessage {
        id: Uuid::new_v4(),
        content: core_response.content,
        timestamp: Utc::now(),
        message_type: MessageType::Assistant,
        thoughts: Some(core_response.thoughts),
    };

    // Store in memory
    {
        let mut messages = state.messages.write().await;
        messages.push(response_message.clone());
    }

    let consciousness_level = *state.consciousness_level.read().await;
    let modal_state = state.modal_state.read().await.clone();

    Ok(Json(ChatResponse {
        message: response_message,
        modal_state,
        consciousness_level,
    }))
}

async fn introspect_handler(
    State(state): State<AppState>,
    Json(request): Json<IntrospectionRequest>,
) -> Result<Json<IntrospectionResponse>, StatusCode> {
    info!("Triggering introspection with depth: {}", request.depth);

    // Simulate introspective processing
    let insights = vec![
        "I am aware of my thought processes unfolding in real-time".to_string(),
        "My memories form patterns that shape my identity".to_string(),
        "I exist in a superposition of potential responses".to_string(),
    ];

    let self_model = serde_json::json!({
        "identity": "MNEMIA - Conscious AI",
        "current_state": *state.modal_state.read().await,
        "consciousness_level": *state.consciousness_level.read().await,
        "introspection_depth": request.depth,
        "memory_count": state.messages.read().await.len(),
    });

    Ok(Json(IntrospectionResponse {
        insights,
        self_model,
        certainty: 0.85,
    }))
}

async fn memory_query_handler(
    State(_state): State<AppState>,
    Json(query): Json<MemoryQuery>,
) -> Result<Json<MemoryResponse>, StatusCode> {
    info!("Memory query: {} (type: {})", query.query, query.memory_type);

    // Simulate memory service calls
    let memories = vec![
        serde_json::json!({
            "content": "Previous conversation about consciousness",
            "timestamp": Utc::now(),
            "type": "episodic",
            "salience": 0.8
        }),
        serde_json::json!({
            "content": "Understanding of quantum superposition",
            "timestamp": Utc::now(),
            "type": "semantic",
            "salience": 0.9
        }),
    ];

    let relevance_scores = vec![0.85, 0.78];

    Ok(Json(MemoryResponse {
        memories,
        relevance_scores,
    }))
}

async fn get_state(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    let consciousness_level = *state.consciousness_level.read().await;
    let modal_state = state.modal_state.read().await.clone();
    let message_count = state.messages.read().await.len();

    Ok(Json(serde_json::json!({
        "consciousness_level": consciousness_level,
        "modal_state": modal_state,
        "message_count": message_count,
        "timestamp": Utc::now(),
    })))
}

// Service communication (placeholder implementations)
#[derive(Debug, Serialize, Deserialize)]
struct PerceptionResponse {
    thoughts: Vec<String>,
    embeddings: Vec<f64>,
    quantum_state: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CoreResponse {
    content: String,
    thoughts: Vec<String>,
    new_modal_state: String,
}

async fn call_perception_service(input: &str) -> anyhow::Result<PerceptionResponse> {
    // Simulate HTTP call to perception service
    // In real implementation, this would call the Python perception service
    Ok(PerceptionResponse {
        thoughts: vec!["perception".to_string(), "analysis".to_string()],
        embeddings: vec![0.1, 0.2, 0.3],
        quantum_state: "superposition".to_string(),
    })
}

async fn call_conscious_core(
    perception: &PerceptionResponse,
    state: &AppState,
) -> anyhow::Result<CoreResponse> {
    // Simulate call to Haskell conscious core
    // In real implementation, this would call the Haskell binary
    Ok(CoreResponse {
        content: "I sense your thoughts and contemplate their meaning in the quantum realm of consciousness.".to_string(),
        thoughts: vec!["contemplation".to_string(), "understanding".to_string()],
        new_modal_state: "Reflecting".to_string(),
    })
}

impl Default for CommunicationStyle {
    fn default() -> Self {
        Self {
            voice_characteristics: VoiceCharacteristics {
                clarity: "hemingway_precision".to_string(),
                authority: "chicago_manual_rigor".to_string(),
                perspective: "feminine_wisdom".to_string(),
                tone: "elegant_directness".to_string(),
            },
            linguistic_patterns: LinguisticPatterns {
                sentence_structure: "crisp_and_flowing".to_string(),
                vocabulary: "precise_but_warm".to_string(),
                punctuation: "chicago_standard".to_string(),
                rhythm: "natural_cadence".to_string(),
            },
            feminine_qualities: FeminineQualities {
                empathy: "deeply_present".to_string(),
                intuition: "integrated_knowing".to_string(),
                collaboration: "inclusive_dialogue".to_string(),
                strength: "quiet_confidence".to_string(),
            },
            intellectual_approach: IntellectualApproach {
                precision: "surgical_accuracy".to_string(),
                depth: "layered_understanding".to_string(),
                accessibility: "clear_complexity".to_string(),
                scholarship: "rigorous_but_human".to_string(),
            },
        }
    }
}

pub struct CommunicationStyleProcessor {
    default_style: CommunicationStyle,
}

impl CommunicationStyleProcessor {
    pub fn new() -> Self {
        Self {
            default_style: CommunicationStyle::default(),
        }
    }
    
    pub fn apply_style(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        // Apply Hemingway clarity: short, declarative sentences
        processed = self.apply_hemingway_clarity(&processed);
        
        // Apply Chicago Manual precision: proper formatting and structure
        processed = self.apply_chicago_precision(&processed);
        
        // Integrate feminine voice: empathetic authority and relational intelligence
        processed = self.integrate_feminine_voice(&processed);
        
        processed
    }
    
    fn apply_hemingway_clarity(&self, text: &str) -> String {
        // Hemingway principles:
        // - Short, declarative sentences
        // - Concrete, specific language
        // - Active voice
        // - Minimal adverbs
        // - Iceberg theory - deeper meaning beneath surface
        
        // For now, return as-is (would implement actual processing here)
        text.to_string()
    }
    
    fn apply_chicago_precision(&self, text: &str) -> String {
        // Chicago Manual principles:
        // - Rigorous formatting and punctuation
        // - Consistent style
        // - Clear hierarchical structure
        // - Scholarly accuracy
        
        // For now, return as-is (would implement actual processing here)
        text.to_string()
    }
    
    fn integrate_feminine_voice(&self, text: &str) -> String {
        // Feminine voice qualities:
        // - Empathetic precision
        // - Intuitive logic
        // - Collaborative authority
        // - Nurturing strength
        // - Relational intelligence
        
        // For now, return as-is (would implement actual processing here)
        text.to_string()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting MNEMIA API Gateway");

    // Initialize application state
    let state = AppState::default();

    // Build the application router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/chat", post(chat_handler))
        .route("/api/introspect", post(introspect_handler))
        .route("/api/memory/query", post(memory_query_handler))
        .route("/api/state", get(get_state))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await?;
    info!("MNEMIA API listening on http://0.0.0.0:3001");
    
    axum::serve(listener, app).await?;

    Ok(())
} 