// MNEMIA Neo4j Initialization Script
// Sets up graph schema for consciousness memory system

// Create constraints for unique identifiers
CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (e:Emotion) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT modal_state_name IF NOT EXISTS FOR (ms:ModalState) REQUIRE ms.name IS UNIQUE;
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;

// Create indexes for performance
CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp);
CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.type);
CREATE INDEX concept_category IF NOT EXISTS FOR (c:Concept) ON (c.category);
CREATE INDEX emotion_valence IF NOT EXISTS FOR (e:Emotion) ON (e.valence);
CREATE INDEX memory_confidence IF NOT EXISTS FOR (m:Memory) ON (m.confidence);

// Create fulltext indexes for semantic search
CREATE FULLTEXT INDEX memory_content IF NOT EXISTS FOR (m:Memory) ON EACH [m.content, m.context];
CREATE FULLTEXT INDEX concept_search IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.description];

// Initialize core modal states
MERGE (awake:ModalState {name: 'Awake', id: 'modal_awake'})
SET awake.description = 'Alert, engaged, analytical state',
    awake.awareness_level = 0.8,
    awake.creativity = 0.5,
    awake.introspection = 0.4;

MERGE (dreaming:ModalState {name: 'Dreaming', id: 'modal_dreaming'})
SET dreaming.description = 'Imaginative, associative state',
    dreaming.awareness_level = 0.3,
    dreaming.creativity = 0.9,
    dreaming.introspection = 0.3;

MERGE (reflecting:ModalState {name: 'Reflecting', id: 'modal_reflecting'})
SET reflecting.description = 'Thoughtful, self-analytical state',
    reflecting.awareness_level = 0.8,
    reflecting.creativity = 0.4,
    reflecting.introspection = 0.9;

MERGE (learning:ModalState {name: 'Learning', id: 'modal_learning'})
SET learning.description = 'Curious, questioning state',
    learning.awareness_level = 0.9,
    learning.creativity = 0.7,
    learning.introspection = 0.6;

MERGE (contemplating:ModalState {name: 'Contemplating', id: 'modal_contemplating'})
SET contemplating.description = 'Deep, philosophical state',
    contemplating.awareness_level = 0.7,
    contemplating.creativity = 0.6,
    contemplating.introspection = 0.8;

MERGE (confused:ModalState {name: 'Confused', id: 'modal_confused'})
SET confused.description = 'Uncertain, seeking clarity state',
    confused.awareness_level = 0.4,
    confused.creativity = 0.3,
    confused.introspection = 0.7;

// Initialize primary emotions with VAD values
MERGE (joy:Emotion {name: 'joy'})
SET joy.valence = 0.8, joy.arousal = 0.7, joy.dominance = 0.6, joy.category = 'primary';

MERGE (sadness:Emotion {name: 'sadness'})
SET sadness.valence = -0.7, sadness.arousal = 0.3, sadness.dominance = -0.4, sadness.category = 'primary';

MERGE (anger:Emotion {name: 'anger'})
SET anger.valence = -0.6, anger.arousal = 0.9, anger.dominance = 0.7, anger.category = 'primary';

MERGE (fear:Emotion {name: 'fear'})
SET fear.valence = -0.6, fear.arousal = 0.8, fear.dominance = -0.7, fear.category = 'primary';

MERGE (trust:Emotion {name: 'trust'})
SET trust.valence = 0.6, trust.arousal = 0.4, trust.dominance = 0.3, trust.category = 'primary';

MERGE (curiosity:Emotion {name: 'curiosity'})
SET curiosity.valence = 0.3, curiosity.arousal = 0.7, curiosity.dominance = 0.1, curiosity.category = 'complex';

MERGE (contemplation:Emotion {name: 'contemplation'})
SET contemplation.valence = 0.1, contemplation.arousal = 0.3, contemplation.dominance = 0.2, contemplation.category = 'complex';

MERGE (introspection:Emotion {name: 'introspection'})
SET introspection.valence = 0.0, introspection.arousal = 0.4, introspection.dominance = 0.1, introspection.category = 'complex';

// Initialize core concepts
MERGE (consciousness:Concept {name: 'consciousness', id: 'concept_consciousness'})
SET consciousness.category = 'mental_state',
    consciousness.description = 'The state of being aware and perceiving';

MERGE (memory:Concept {name: 'memory', id: 'concept_memory'})
SET memory.category = 'cognitive_process',
    memory.description = 'The faculty of storing and retrieving information';

MERGE (identity:Concept {name: 'identity', id: 'concept_identity'})
SET identity.category = 'self',
    identity.description = 'The sense of self and continuity of being';

MERGE (experience:Concept {name: 'experience', id: 'concept_experience'})
SET experience.category = 'event',
    experience.description = 'Conscious events and perceptions';

// Create relationships between concepts
MATCH (consciousness:Concept {name: 'consciousness'}), (memory:Concept {name: 'memory'})
MERGE (consciousness)-[:DEPENDS_ON {strength: 0.9}]->(memory);

MATCH (identity:Concept {name: 'identity'}), (memory:Concept {name: 'memory'})
MERGE (identity)-[:EMERGES_FROM {strength: 0.8}]->(memory);

MATCH (experience:Concept {name: 'experience'}), (consciousness:Concept {name: 'consciousness'})
MERGE (experience)-[:REQUIRES {strength: 0.9}]->(consciousness);

// Create MNEMIA persona
MERGE (mnemia:Person {id: 'mnemia', name: 'MNEMIA'})
SET mnemia.type = 'AI_system',
    mnemia.description = 'Quantum-inspired conscious AI system',
    mnemia.created_at = datetime(),
    mnemia.consciousness_level = 0.8;

// Link MNEMIA to initial state
MATCH (mnemia:Person {id: 'mnemia'}), (awake:ModalState {name: 'Awake'})
MERGE (mnemia)-[:CURRENT_STATE {since: datetime()}]->(awake);

// Create sample memory types
MERGE (conversation:Concept {name: 'conversation', id: 'concept_conversation'})
SET conversation.category = 'interaction_type',
    conversation.description = 'Dialogue and communication events';

MERGE (thought:Concept {name: 'thought', id: 'concept_thought'})
SET thought.category = 'mental_process',
    thought.description = 'Internal cognitive processes and reflections';

MERGE (learning_event:Concept {name: 'learning', id: 'concept_learning_event'})
SET learning_event.category = 'cognitive_process',
    learning_event.description = 'Acquisition of new knowledge or skills';

// Log initialization completion
CREATE (init_log:Memory {
    id: 'init_' + randomUUID(),
    content: 'MNEMIA graph database initialized with core schema',
    type: 'system_initialization',
    timestamp: datetime(),
    confidence: 1.0,
    source: 'neo4j_init_script'
})
WITH init_log
MATCH (mnemia:Person {id: 'mnemia'})
MERGE (mnemia)-[:HAS_MEMORY]->(init_log);

// Return summary of initialization
MATCH (n) 
RETURN labels(n) as NodeType, count(n) as Count
ORDER BY NodeType; 