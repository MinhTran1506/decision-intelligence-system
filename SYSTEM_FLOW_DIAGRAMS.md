# Decision Intelligence Studio - System Flow Diagrams

Complete visual guide to how the system works.

---

## 📊 DIAGRAM 1: Overall System Architecture

```mermaid
graph TB
    subgraph "User Interfaces"
        UI1[Streamlit App<br/>Port 8501]
        UI2[HTML Dashboard<br/>Port 8080]
        UI3[API Docs<br/>Port 8001/docs]
    end
    
    subgraph "API Layer"
        API1[Enhanced API<br/>FastAPI + WebSocket<br/>Port 8001]
        API2[Original API<br/>FastAPI<br/>Port 8000]
    end
    
    subgraph "Processing Layer"
        RT[Real-time Processor<br/>Event Streaming]
        BATCH[Batch Pipeline<br/>Scheduled Jobs]
    end
    
    subgraph "ML Layer"
        CAUSAL[Causal Estimation<br/>DoWhy + EconML]
        REF[Refutation Tests<br/>Validation]
        DQ[Data Quality<br/>Checks]
    end
    
    subgraph "Data Layer"
        RAW[Raw Data<br/>marketing_events.parquet]
        CAN[Canonical Data<br/>canonical_events.parquet]
        UPLIFT[Uplift Scores<br/>uplift_scores.parquet]
        MODEL[Model Artifacts<br/>causal_forest_v1.joblib]
    end
    
    UI1 --> API1
    UI2 --> API1
    UI3 --> API1
    UI1 --> API2
    
    API1 --> UPLIFT
    API1 --> MODEL
    API2 --> UPLIFT
    
    RT --> API1
    BATCH --> DQ
    DQ --> CAN
    CAN --> CAUSAL
    CAUSAL --> REF
    REF --> UPLIFT
    CAUSAL --> MODEL
    
    RAW --> BATCH
    
    style UI1 fill:#667eea,color:#fff
    style API1 fill:#764ba2,color:#fff
    style CAUSAL fill:#ed64a6,color:#fff
    style UPLIFT fill:#4ecdc4,color:#fff
```

**Key Points:**
- **3 User Interfaces** → All powered by APIs
- **2 API Services** → Enhanced (WebSocket) + Original (REST)
- **2 Processing Modes** → Real-time streaming + Batch jobs
- **3 ML Steps** → Quality checks → Causal estimation → Validation
- **4 Data Stages** → Raw → Canonical → Scores → Models

---

## 📊 DIAGRAM 2: Data Pipeline Flow (Batch Processing)

```mermaid
flowchart TD
    START([Start Pipeline]) --> GEN[Generate Sample Data<br/>10,000 customers<br/>~2 seconds]
    GEN --> SAVE1[(Save to:<br/>data/raw/marketing_events.parquet)]
    
    SAVE1 --> ETL[Create Canonical Dataset<br/>Transform to standard schema<br/>~1 second]
    ETL --> SAVE2[(Save to:<br/>data/processed/canonical_events.parquet)]
    
    SAVE2 --> DQ{Data Quality<br/>Checks}
    DQ -->|FAIL| ALERT1[Send Alert &<br/>Stop Pipeline]
    DQ -->|PASS| CAUSAL[Causal Estimation<br/>DoWhy Identification<br/>EconML CausalForestDML<br/>~60 seconds]
    
    CAUSAL --> SPLIT[Split Results]
    SPLIT --> SCORES[Uplift Scores<br/>Per Customer]
    SPLIT --> MODEL[Trained Model<br/>CausalForestDML]
    
    SCORES --> SAVE3[(Save to:<br/>data/outputs/uplift_scores.parquet)]
    MODEL --> SAVE4[(Save to:<br/>models/causal_forest_v1.joblib)]
    
    SAVE3 --> REF[Refutation Tests<br/>5 Tests<br/>~30 seconds]
    SAVE4 --> REF
    
    REF --> CHECK{All Tests<br/>Pass?}
    CHECK -->|FAIL| ALERT2[Warning Alert<br/>Continue with Caution]
    CHECK -->|PASS| SUCCESS[Pipeline Complete!]
    
    ALERT2 --> REPORT1[Save Report]
    SUCCESS --> REPORT2[Save Report]
    
    REPORT1 --> END([End])
    REPORT2 --> END
    
    style START fill:#4ecdc4
    style SUCCESS fill:#52c41a
    style ALERT1 fill:#ff4d4f
    style ALERT2 fill:#faad14
    style DQ fill:#1890ff
    style CHECK fill:#1890ff
    style CAUSAL fill:#722ed1
    style END fill:#4ecdc4
```

**Timeline:**
- **Total Duration**: ~100 seconds (with refutation tests)
- **Quick Mode**: ~70 seconds (skip refutation)
- **Critical Path**: Data Gen → ETL → DQ → Causal → Save

**Key Decision Points:**
1. **Data Quality Check** - Fails if data issues detected
2. **Refutation Tests** - Warns but continues if tests fail

---

## 📊 DIAGRAM 3: Real-Time Processing Flow

```mermaid
sequenceDiagram
    participant EVENT as Event Source<br/>(Simulated Stream)
    participant GEN as Event Generator
    participant SCORE as Real-time Scorer
    participant ALERT as Alert Manager
    participant WS as WebSocket Manager
    participant CLIENT as Connected Clients<br/>(Browsers)
    
    EVENT->>GEN: Generate Customer Event
    activate GEN
    Note over GEN: Create event with:<br/>- User ID<br/>- Age, Income, Engagement<br/>- Timestamp
    GEN->>SCORE: Send Event
    deactivate GEN
    
    activate SCORE
    SCORE->>SCORE: Extract Features
    SCORE->>SCORE: Load Model<br/>(if not cached)
    SCORE->>SCORE: Compute Uplift Score
    SCORE->>SCORE: Assign Segment
    SCORE->>SCORE: Generate Recommendation
    Note over SCORE: Scoring takes ~45ms
    SCORE->>ALERT: Check Thresholds
    deactivate SCORE
    
    activate ALERT
    ALERT->>ALERT: High Uplift?<br/>(>$80)
    ALERT->>ALERT: Low Treatment Rate?<br/>(<5%)
    alt Alert Triggered
        ALERT->>WS: Send Alert Message
    end
    ALERT->>WS: Send Scored Event
    deactivate ALERT
    
    activate WS
    WS->>CLIENT: Broadcast via WebSocket
    Note over WS,CLIENT: <10ms latency
    deactivate WS
    
    CLIENT->>CLIENT: Update Dashboard<br/>in Real-time
    
    Note over EVENT,CLIENT: Process repeats at<br/>2-5 events per second
```

**Performance:**
- **Event Generation**: Continuous at 2-5 events/sec
- **Scoring Latency**: ~45ms per event
- **WebSocket Broadcast**: <10ms to all clients
- **Total E2E Latency**: ~55ms (event → dashboard)

---

## 📊 DIAGRAM 4: User Journey Through Streamlit App

```mermaid
flowchart TD
    START([User Opens<br/>localhost:8501]) --> LOAD[Load Streamlit App]
    LOAD --> CACHE{Data<br/>Cached?}
    
    CACHE -->|NO| READ[Read Data Files:<br/>- uplift_scores.parquet<br/>- canonical_events.parquet<br/>- reports JSON]
    CACHE -->|YES| OVERVIEW
    READ --> OVERVIEW
    
    OVERVIEW[📊 Overview Dashboard<br/>Default Landing Page]
    
    OVERVIEW --> NAV{User<br/>Navigation<br/>Choice}
    
    NAV -->|Click Real-time| RT[📡 Real-time Monitoring]
    NAV -->|Click A/B Test| AB[🧪 A/B Test Tracking]
    NAV -->|Click Customer| CUSTOMER[🔍 Customer Lookup]
    NAV -->|Click Models| MODEL[🔬 Model Comparison]
    NAV -->|Click Registry| REGISTRY[📦 Model Registry]
    NAV -->|Click Analytics| ANALYTICS[📈 Advanced Analytics]
    NAV -->|Stay| OVERVIEW
    
    RT --> INTERACT1{User<br/>Interaction}
    AB --> INTERACT2{User<br/>Interaction}
    CUSTOMER --> INTERACT3{User<br/>Interaction}
    MODEL --> INTERACT4{User<br/>Interaction}
    REGISTRY --> INTERACT5{User<br/>Interaction}
    ANALYTICS --> INTERACT6{User<br/>Interaction}
    
    INTERACT1 -->|View Metrics| DISPLAY1[Show Live Stats]
    INTERACT1 -->|View Trends| TRENDS1[Line Charts Over Time]
    INTERACT1 -->|Simulate Events| SIM1[Generate Test Events]
    
    INTERACT2 -->|Create Test| CREATE2[Setup A/B Test]
    INTERACT2 -->|View Results| RESULTS2[Show Test Outcomes]
    INTERACT2 -->|Complete Test| COMPLETE2[Declare Winner]
    
    INTERACT3 -->|Search User| SEARCH3[Find Customer]
    INTERACT3 -->|View Profile| PROFILE3[Show CATE & Details]
    INTERACT3 -->|Get Recommendation| REC3[Treat/Don't Treat]
    
    INTERACT4 -->|View ATE| ATE4[Show Estimates]
    INTERACT4 -->|Check Refutation| REF4[5 Validation Tests]
    INTERACT4 -->|Compare Models| COMPARE4[Side-by-Side]
    
    INTERACT5 -->|View Production| PROD5[Current Model]
    INTERACT5 -->|List Versions| VERSIONS5[All Models]
    INTERACT5 -->|Check Drift| DRIFT5[Drift Detection]
    
    INTERACT6 -->|Feature Importance| IMPORTANCE6[Top Features]
    INTERACT6 -->|Segment Analysis| SEGMENT6[By Customer Group]
    INTERACT6 -->|What-If| WHATIF6[Scenario Analysis]
    
    DISPLAY1 --> NAV
    TRENDS1 --> NAV
    SIM1 --> NAV
    CREATE2 --> NAV
    RESULTS2 --> NAV
    COMPLETE2 --> NAV
    SEARCH3 --> NAV
    PROFILE3 --> NAV
    REC3 --> NAV
    ATE4 --> NAV
    REF4 --> NAV
    COMPARE4 --> NAV
    PROD5 --> NAV
    VERSIONS5 --> NAV
    DRIFT5 --> NAV
    IMPORTANCE6 --> NAV
    SEGMENT6 --> NAV
    WHATIF6 --> NAV
    
    NAV -->|Exit| END([Close App])
    
    style START fill:#4ecdc4
    style OVERVIEW fill:#667eea,color:#fff
    style RT fill:#764ba2,color:#fff
    style AB fill:#ed64a6,color:#fff
    style CUSTOMER fill:#f093fb,color:#fff
    style MODEL fill:#ff9a9e,color:#fff
    style REGISTRY fill:#a8edea,color:#333
    style ANALYTICS fill:#feca57,color:#333
    style END fill:#4ecdc4
```

**User Flow:**
1. **Entry** → Streamlit loads data (2 sec first time, instant after caching)
2. **Default** → Overview Dashboard with key metrics
3. **Navigation** → 7 specialized pages via sidebar
4. **Interaction** → Charts, filters, hover effects, tables
5. **Return** → Easy navigation between pages

**Pages:**
- **Overview** - KPIs, distributions, segment performance
- **Real-time Monitoring** - Live event scoring with trend analytics
- **A/B Test Tracking** - Create, monitor, and complete experiments
- **Customer Lookup** - Individual customer CATE predictions
- **Model Comparison** - ATE estimates and refutation tests
- **Model Registry** - Model versioning, promotion, drift detection
- **Advanced Analytics** - Feature importance, segment deep-dives

---

## 📊 DIAGRAM 5: API Request/Response Flow

```mermaid
sequenceDiagram
    participant CLIENT as Client Application<br/>(Browser/Script)
    participant API as FastAPI Server<br/>Port 8001
    participant LOAD as Model Loader
    participant MODEL as Trained Model<br/>(in memory)
    participant DATA as Data Storage<br/>(Parquet files)
    participant WS as WebSocket Manager
    
    Note over CLIENT,DATA: === REST API Flow ===
    
    CLIENT->>API: POST /batch-score<br/>{users: [...]}
    activate API
    API->>API: Parse Request<br/>Validate JSON
    API->>LOAD: Load Model<br/>(if not in memory)
    activate LOAD
    LOAD->>MODEL: Get Model Object
    deactivate LOAD
    API->>MODEL: model.effect(X)
    activate MODEL
    MODEL->>MODEL: Predict Uplift<br/>for each user
    MODEL->>API: Return Scores
    deactivate MODEL
    API->>API: Apply Business Rules:<br/>- Calculate ROI<br/>- Assign Segments<br/>- Generate Recommendations
    API->>CLIENT: Response:<br/>{results: [...]}
    deactivate API
    Note over CLIENT,API: Typical latency: 50-100ms
    
    Note over CLIENT,DATA: === WebSocket Flow ===
    
    CLIENT->>API: WebSocket Connect<br/>ws://localhost:8001/ws
    activate API
    API->>WS: Register Client
    activate WS
    WS->>CLIENT: Connection Established
    Note over WS,CLIENT: Connection kept open
    
    loop Every 2-5 seconds
        API->>API: New Event Generated
        API->>MODEL: Score Event
        MODEL->>API: Return Score
        API->>WS: Broadcast Event
        WS->>CLIENT: Send JSON Message
        Note over WS,CLIENT: <10ms latency
    end
    
    CLIENT->>API: Close Connection
    API->>WS: Unregister Client
    deactivate WS
    deactivate API
    
    Note over CLIENT,DATA: === Data Query Flow ===
    
    CLIENT->>API: GET /stats
    activate API
    API->>DATA: Read uplift_scores.parquet
    activate DATA
    DATA->>API: Return DataFrame
    deactivate DATA
    API->>API: Compute Statistics:<br/>- Segment aggregations<br/>- Treatment rates<br/>- Outcome means
    API->>CLIENT: Response:<br/>{stats: {...}}
    deactivate API
```

**API Patterns:**
1. **Synchronous REST** - Request → Process → Response
2. **Asynchronous WebSocket** - Connect → Stream → Close
3. **Data Caching** - Model loaded once, reused
4. **Error Handling** - Validation at entry, graceful failures

---

## 📊 DIAGRAM 6: Causal Estimation Deep Dive

```mermaid
flowchart TB
    START([Canonical Dataset<br/>Ready]) --> SPLIT[Split into:<br/>X Features, T Treatment, Y Outcome]
    
    SPLIT --> GRAPH[Build Causal Graph<br/>from Config]
    GRAPH --> NODES[Define Nodes:<br/>age, income, past_purchases,<br/>engagement, treatment, outcome]
    NODES --> EDGES[Define Edges:<br/>Confounders → Treatment<br/>Confounders → Outcome<br/>Treatment → Outcome]
    EDGES --> DOWHY[DoWhy CausalModel<br/>Identification]
    
    DOWHY --> IDENTIFY{Causal Effect<br/>Identifiable?}
    IDENTIFY -->|NO| FAIL[Error: Not Identifiable<br/>Check assumptions]
    IDENTIFY -->|YES| ESTIMAND[Identified Estimand:<br/>Backdoor Adjustment]
    
    ESTIMAND --> ATE[Estimate ATE<br/>Linear Regression<br/>with backdoor adjustment]
    ATE --> ATE_VALUE[ATE ≈ $45.23]
    
    ATE_VALUE --> CATE_PREP[Prepare for CATE<br/>Estimation]
    CATE_PREP --> BASE_MODELS[Train Base Models]
    
    BASE_MODELS --> MODEL_Y[Model Y:<br/>RandomForest<br/>Predicts Outcome<br/>200 trees]
    BASE_MODELS --> MODEL_T[Model T:<br/>RandomForest<br/>Predicts Treatment<br/>200 trees]
    
    MODEL_Y --> DML[CausalForestDML<br/>Double Machine Learning]
    MODEL_T --> DML
    
    DML --> CROSSFIT[Cross-fitting<br/>2 folds<br/>Reduces bias]
    CROSSFIT --> FOREST[Causal Forest<br/>1000 trees<br/>Captures heterogeneity]
    
    FOREST --> PREDICT[Predict Individual<br/>Treatment Effects<br/>for each customer]
    
    PREDICT --> CATE_DIST[CATE Distribution:<br/>min=$8.73<br/>mean=$45.12<br/>max=$98.45<br/>std=$18.45]
    
    CATE_DIST --> SEGMENT[Segment Customers<br/>by Uplift Quantiles]
    SEGMENT --> SEG1[High Uplift 25%<br/>$70-98]
    SEGMENT --> SEG2[Medium-High 25%<br/>$45-70]
    SEGMENT --> SEG3[Medium-Low 25%<br/>$20-45]
    SEGMENT --> SEG4[Low Uplift 25%<br/>$0-20]
    
    SEG1 --> SAVE
    SEG2 --> SAVE
    SEG3 --> SAVE
    SEG4 --> SAVE
    
    SAVE[Save Results:<br/>uplift_scores.parquet<br/>model artifact .joblib]
    
    SAVE --> REF[Refutation Tests]
    
    REF --> TEST1[1. Placebo Test<br/>Shuffle treatment,<br/>expect ~0 effect]
    REF --> TEST2[2. Random Cause<br/>Add random variable,<br/>check stability]
    REF --> TEST3[3. Subset Validation<br/>Estimate on subset,<br/>compare]
    REF --> TEST4[4. Data Subset<br/>Use 80% of data,<br/>check consistency]
    REF --> TEST5[5. Bootstrap<br/>Resample 50 times,<br/>check CI]
    
    TEST1 --> RESULTS
    TEST2 --> RESULTS
    TEST3 --> RESULTS
    TEST4 --> RESULTS
    TEST5 --> RESULTS
    
    RESULTS[Refutation Results:<br/>Pass Rate: 100%<br/>All tests passed] --> END([Estimation Complete<br/>Ready for Production])
    
    FAIL --> END
    
    style START fill:#4ecdc4
    style DOWHY fill:#667eea,color:#fff
    style DML fill:#764ba2,color:#fff
    style FOREST fill:#ed64a6,color:#fff
    style REF fill:#ff9a9e,color:#fff
    style END fill:#52c41a
    style FAIL fill:#ff4d4f
```

**Estimation Steps:**
1. **Identification** (DoWhy) - Prove effect is estimable
2. **ATE Estimation** - Linear regression with adjustment
3. **Base Models** - RF for outcome and treatment
4. **CATE Estimation** - CausalForestDML with 1000 trees
5. **Segmentation** - Quantile-based bucketing
6. **Validation** - 5 refutation tests

**Key Output:**
- Individual uplift scores for 10,000 customers
- Trained model artifact (20MB)
- Validation report (all tests passed)

---

## 📊 DIAGRAM 7: Model Serving Architecture

```mermaid
graph TB
    subgraph "Request Entry Points"
        HTTP[HTTP Request<br/>REST API]
        WS[WebSocket<br/>Connection]
        BATCH[Batch Job<br/>Scheduled]
    end
    
    subgraph "Load Balancer Layer"
        LB[Load Balancer<br/>Nginx/Cloud LB]
    end
    
    subgraph "API Instance 1"
        API1[FastAPI Server 1]
        CACHE1[Model Cache]
        MODEL1[Loaded Model]
    end
    
    subgraph "API Instance 2"
        API2[FastAPI Server 2]
        CACHE2[Model Cache]
        MODEL2[Loaded Model]
    end
    
    subgraph "API Instance N"
        APIN[FastAPI Server N]
        CACHE3[Model Cache]
        MODEL3[Loaded Model]
    end
    
    subgraph "Shared Storage"
        S3[Cloud Storage<br/>S3/GCS/Azure Blob]
        MODELS[(Model Registry<br/>models/)]
        DATA[(Data Lake<br/>data/)]
    end
    
    subgraph "Monitoring"
        METRICS[Prometheus<br/>Metrics]
        LOGS[Centralized Logs<br/>ELK/Cloud Logging]
        ALERTS[Alert Manager<br/>PagerDuty/Slack]
    end
    
    HTTP --> LB
    WS --> LB
    
    LB --> API1
    LB --> API2
    LB --> APIN
    
    API1 --> CACHE1
    CACHE1 --> MODEL1
    API2 --> CACHE2
    CACHE2 --> MODEL2
    APIN --> CACHE3
    CACHE3 --> MODEL3
    
    MODEL1 --> S3
    MODEL2 --> S3
    MODEL3 --> S3
    
    BATCH --> MODELS
    MODELS --> S3
    DATA --> S3
    
    API1 --> METRICS
    API2 --> METRICS
    APIN --> METRICS
    
    API1 --> LOGS
    API2 --> LOGS
    APIN --> LOGS
    
    METRICS --> ALERTS
    LOGS --> ALERTS
    
    style HTTP fill:#4ecdc4
    style LB fill:#667eea,color:#fff
    style API1 fill:#764ba2,color:#fff
    style S3 fill:#ff9a9e
    style METRICS fill:#52c41a
```

**Deployment Architecture:**
- **Horizontal Scaling** - Multiple API instances
- **Load Balancing** - Distribute requests
- **Model Caching** - Load once per instance
- **Shared Storage** - Centralized model registry
- **Monitoring** - Metrics, logs, alerts

---

## 📊 DIAGRAM 8: Demo Navigation Map

```mermaid
mindmap
  root((Decision Intelligence<br/>Studio Demo))
    Overview Dashboard
      Key Metrics
        ATE: $45
        Total Customers
        Treatment Rate
        High-Uplift Count
      Uplift Distribution
        Violin Plot
        By Segment
      Segment Performance
        Table with ROI
        Revenue calculations
      Key Insights
        Heterogeneity
        ROI Optimization
        Precision Targeting
    Real-time Monitoring
      Live Metrics
        Events/sec
        Latency
        High-Value %
        Active Alerts
      Event Stream Chart
        Moving average
        Time series
      Event History
        Trends Over Time
        Distribution Analysis
        Segment Performance
      Recent Events Table
        Last 20 scored
        User details
      Alert Panel
        Warnings
        Info messages
    A/B Test Tracking
      Test Results Table
        Predicted vs Observed
        Error metrics
      Calibration Plot
        Scatter diagram
        Perfect line
      Test Summary
        Completed count
        Avg calibration
      Next Steps
        Recommendations
    Customer Lookup
      Search Interface
        User ID input
        Search button
      Customer Profile
        Demographics
        Historical data
      CATE Prediction
        Individual uplift
        Confidence interval
      Treatment Recommendation
        Treat/Don't Treat
        Expected value
    Model Comparison
      Version Table
        All models
        Performance metrics
      ATE Comparison
        Bar chart
      Refutation Results
        5 Validation Tests
        Pass/Fail status
      Performance vs Complexity
        Scatter plot
      Recommendation
        Best model selection
    Model Registry
      Production Model
        Current version
        Deployment date
      Model Versions
        Version history
        Performance metrics
      Drift Detection
        Distribution shift
        Alerts
      Promotion Actions
        Deploy to production
        Rollback
    Advanced Analytics
      Feature Importance
        Bar chart
        Top 5 features
      Interaction Effects
        Heatmap
        Age × Income
      Uplift vs Engagement
        Scatter plot
        Trend line
      Temporal Analysis
        Time series
        Monthly averages
    API Demonstration
      OpenAPI Docs
        20+ endpoints
      Batch Scoring
        Live demo
        Request/Response
      Feature Importance
        GET endpoint
      WebSocket
        Browser console
        Live connection
```

**Navigation Flow:**
- Start at center (Overview)
- Branch to any of 7 specialized pages
- Each page has 3-5 sub-sections
- Can navigate freely between all pages

---

## 📊 DIAGRAM 9: Data Flow Timeline

```mermaid
gantt
    title Pipeline Execution Timeline (Quick Mode ~70 seconds)
    dateFormat ss
    axisFormat %S
    
    section Data Generation
    Generate Sample Data           :gen, 00, 2s
    Save to Parquet                :save1, after gen, 1s
    
    section ETL
    Load Raw Data                  :load, after save1, 1s
    Create Canonical Schema        :etl, after load, 1s
    Save Canonical Data            :save2, after etl, 1s
    
    section Quality
    Load Canonical Data            :loadc, after save2, 1s
    Run DQ Checks                  :dq, after loadc, 3s
    Save DQ Report                 :savedq, after dq, 1s
    
    section Causal ML
    Prepare Data                   :prep, after savedq, 2s
    DoWhy Identification          :dowhy, after prep, 3s
    Estimate ATE                   :ate, after dowhy, 2s
    Train Base Models             :base, after ate, 10s
    CausalForestDML               :forest, after base, 45s
    Calculate Uplift Scores       :uplift, after forest, 2s
    
    section Output
    Save Uplift Scores            :scoressave, after uplift, 1s
    Save Model Artifact           :modelsave, after uplift, 2s
    
    section Validation
    Load Results                  :loadres, after modelsave, 1s
    (Optional: Refutation Tests)  :ref, after loadres, 30s
```

**Critical Path:**
- **Fast Steps** (1-3s): Data I/O, ETL, DQ checks
- **Slow Steps** (10-45s): Base model training, Causal Forest
- **Bottleneck**: CausalForestDML with 1000 trees (45s)
- **Total**: ~70s (quick mode) or ~100s (with refutation)

---

## 📊 DIAGRAM 10: Business Value Flow

```mermaid
flowchart LR
    START([Business Problem:<br/>Marketing Waste]) --> COLLECT[Collect Customer Data]
    
    COLLECT --> SYSTEM[Decision Intelligence<br/>System]
    
    SYSTEM --> PROCESS1[Causal Estimation]
    PROCESS1 --> OUTPUT1[Individual Uplift Scores]
    
    OUTPUT1 --> SEGMENT[Segment Customers]
    SEGMENT --> HIGH[High Uplift<br/>$70-98 each<br/>795% ROI]
    SEGMENT --> MED[Medium Uplift<br/>$20-70 each<br/>300% ROI]
    SEGMENT --> LOW[Low Uplift<br/>$0-20 each<br/>50% ROI]
    
    HIGH --> DECISION{Business Decision}
    MED --> DECISION
    LOW --> DECISION
    
    DECISION --> TREAT[Target High & Medium<br/>with promotions]
    DECISION --> NOTREAT[Don't waste budget<br/>on Low segment]
    
    TREAT --> REVENUE[Incremental Revenue:<br/>$445K extra profit<br/>per $100K spent]
    
    REVENUE --> VALIDATE[Run A/B Test<br/>to Validate]
    VALIDATE --> FEEDBACK{Predictions<br/>Accurate?}
    
    FEEDBACK -->|YES| SCALE[Scale to Full<br/>Production]
    FEEDBACK -->|NO| RETRAIN[Retrain Model<br/>with New Data]
    
    RETRAIN --> SYSTEM
    SCALE --> SUCCESS([Business Value:<br/>4x Better ROI<br/>Continuous Improvement])
    
    style START fill:#ff4d4f
    style SYSTEM fill:#667eea,color:#fff
    style HIGH fill:#52c41a
    style REVENUE fill:#52c41a
    style SUCCESS fill:#52c41a
```

**Value Chain:**
1. **Problem** → Marketing inefficiency
2. **Solution** → Causal ML system
3. **Output** → Segment-specific strategies
4. **Decision** → Precision targeting
5. **Result** → 4x better ROI
6. **Validation** → A/B tests confirm
7. **Improvement** → Continuous learning loop
