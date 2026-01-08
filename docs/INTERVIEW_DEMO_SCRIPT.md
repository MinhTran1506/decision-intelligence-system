# Decision Intelligence Studio - Interview Demo Script

## 10-Minute Presentation for Joon Solutions Interview

**Demo Title:** "Decision Intelligence Studio: From Dashboards to Causal AI-Powered Decisions"

**Tagline:** *Aligning perfectly with Joon Solutions' mission: "Modern Data Analytics Today, AI-Powered Decisions Tomorrow"*

---

## 🎯 Pre-Demo Checklist

Before you start, ensure:
- [ ] Streamlit app is running: `streamlit run src/streamlit_app/app.py`
- [ ] Data is generated: `python run_pipeline.py` (if not already done)
- [ ] Browser tabs ready:
  1. Streamlit App (http://localhost:8501)
  2. System Flow Diagrams (SYSTEM_FLOW_DIAGRAMS.md preview)
  3. FastAPI Docs (http://localhost:8000/docs) - optional
- [ ] Simulate ~50 events for better visualizations (click "Simulate 10 Events" 5x in Real-time Monitoring)

---

## PART 1: The Problem (1.5 minutes)

### [SLIDE/VERBAL - No screen needed]

**Opening Hook:**
> "Imagine you're a Marketing Director at a retail company. You have a $1 million marketing budget, and you're about to launch a promotional campaign. The traditional approach? Blast everyone with promotions and hope for the best. But here's the problem..."

**The Business Problem:**
> "Studies show that 60-70% of marketing spend is wasted on customers who would have purchased anyway, or worse, on customers who will never respond regardless of what you do. That's $600K-700K potentially wasted."

**The Real Question:**
> "The real question isn't 'Did sales go up after our campaign?' It's 'Which specific customers actually changed their behavior BECAUSE of our campaign?' This is the difference between correlation and causation."

**Connect to Joon Solutions:**
> "This is exactly what Joon Solutions means by going 'beyond dashboards' - moving from descriptive analytics that tell you WHAT happened, to prescriptive analytics that tell you WHO to target and WHY."

---

## PART 2: The Solution Overview (1 minute)

### [SHOW: Diagram 1 - System Architecture OR Diagram 10 - Business Value Flow]

**Navigate to SYSTEM_FLOW_DIAGRAMS.md, show Diagram 10 (Business Value Flow)**

> "Let me show you the Decision Intelligence Studio - an end-to-end system I built that solves this exact problem using Causal AI."

**Walk through the flow:**
> "We start with raw customer data, pass it through a causal estimation engine using DoWhy and EconML, and output individual uplift scores for each customer. These scores tell us exactly how much ADDITIONAL revenue each customer will generate IF we target them.

> The key insight: We can then segment customers into High, Medium, and Low uplift groups. The High uplift segment shows 795% ROI, while the Low segment shows only 50% ROI. By focusing our budget on high-uplift customers, we achieve 4x better marketing efficiency."

---

## PART 3: Live Demo - The Dashboard (5 minutes)

### STEP 3.1: Overview Dashboard (1 minute)

**[SHOW: Streamlit App → Overview page]**

> "Let's dive into the live system. This is the main dashboard, similar to what a business user at Joon Solutions' clients would see."

**Point out key metrics:**
- **ATE (Average Treatment Effect):** "This is the average causal impact of our marketing - approximately $46 per customer. This isn't correlation - this is the TRUE causal effect."
- **ROI Percentage:** "With our targeting strategy, we're seeing ~700% ROI on marketing spend."
- **Treatment Effect Distribution chart:** "This shows heterogeneity - not all customers respond equally. This distribution is key to smart targeting."

**Key talking point:**
> "Notice how we're not just showing 'average' metrics. We're showing the DISTRIBUTION of treatment effects. Some customers have $70+ uplift, others have near-zero. Traditional dashboards miss this entirely."

---

### STEP 3.2: Real-time Monitoring (1.5 minutes)

**[NAVIGATE: Sidebar → Real-time Monitoring]**

> "Now let's see the system in action. This is real-time scoring - as new customers interact with our system, they're instantly scored."

**Click "Simulate 10 Events" button**

> "Watch as events stream in... Each customer is scored in ~45 milliseconds with their individual uplift prediction."

**Point out the Event History Analytics tabs:**

**Tab 1 - Trends Over Time:**
> "This line chart shows our average uplift score trending over time, with confidence bands. The treatment threshold line at $30 helps us make instant decisions - anyone above this line is worth targeting."

**Tab 2 - Distribution Analysis:**
> "The histogram shows our uplift distribution across all scored customers. The donut chart shows our recommendation breakdown - what percentage we're recommending to treat vs. not treat."

**Tab 3 - Segment Performance:**
> "This breaks down performance by customer segment. We can see which segments have the highest potential ROI."

**Key talking point:**
> "This is the bridge between batch ML and real-time decisioning - exactly what Joon Solutions describes as 'AI-powered decisions tomorrow'. The model was trained once, but it's making personalized decisions thousands of times per day."

---

### STEP 3.3: Model Comparison & Refutation (1.5 minutes)

**[NAVIGATE: Sidebar → Model Comparison]**

> "Here's where the data science rigor comes in. This page shows our model validation - crucial for building trust with stakeholders."

**Walk through:**
1. **Current Model Card:** "Our CausalForest model with its ATE estimate and confidence metrics."
2. **Refutation Tests Section:** 
   > "This is critical - we run 5 different refutation tests to validate our causal claims:
   > - **Placebo Test**: If we shuffle treatment randomly, does the effect disappear? ✓
   > - **Random Cause**: If we add random noise, does our estimate stay stable? ✓
   > - **Subset Validation**: Does the effect hold on different data splits? ✓
   > - **Bootstrap**: What are our confidence intervals?
   
   > All 5 tests pass with 100% pass rate. This isn't just predictive accuracy - this is CAUSAL validity."

**Connect to Joon:**
> "This is what separates toy ML projects from production-ready systems. At Joon Solutions, clients need to trust the recommendations. These refutation tests provide that statistical foundation."

---

### STEP 3.4: A/B Test Tracking (1 minute)

**[NAVIGATE: Sidebar → A/B Test Tracking]**

> "Once we deploy recommendations, how do we know they work? This A/B Test Tracking module closes the loop."

**Show:**
- **Create Test tab:** "We can set up experiments directly from the dashboard."
- **Active Tests tab:** "Track ongoing experiments with statistical significance."
- **View Results:** "Compare control vs. treatment with visual charts and confidence metrics."

**Key insight:**
> "The system doesn't just make predictions - it validates them through experimentation. This creates a continuous learning loop: Predict → Deploy → Test → Improve."

---

## PART 4: Technical Deep Dive (2 minutes)

### [SHOW: Diagram 6 - Causal Estimation Deep Dive OR Diagram 2 - Data Pipeline Flow]

**Navigate to SYSTEM_FLOW_DIAGRAMS.md, show Diagram 6**

> "Let me quickly walk through the technical architecture for those interested in the 'how'."

**Key Components:**

1. **DoWhy for Causal Identification:**
   > "We first use DoWhy to build a causal graph and identify whether the causal effect is even estimable. This prevents us from making false causal claims."

2. **EconML CausalForestDML:**
   > "For heterogeneous treatment effect estimation, we use Microsoft's EconML library with CausalForestDML - a Double Machine Learning approach that handles high-dimensional confounders while providing valid confidence intervals."

3. **Refutation Framework:**
   > "Five statistical tests validate our causal assumptions - placebo, random cause, subset validation, data subset, and bootstrap."

**Tech Stack (brief):**
> "The stack includes:
> - **Python** with DoWhy, EconML, scikit-learn
> - **Streamlit** for the interactive dashboard
> - **FastAPI** for the scoring API
> - **Parquet** for efficient data storage
> - Designed for cloud deployment on GCP/AWS with Airflow orchestration"

**Connect to Joon:**
> "This architecture aligns with Joon's modern data stack philosophy - cloud-native, modular, and built for scale. The same patterns work whether you're on BigQuery, Snowflake, or any modern data platform."

---

## PART 5: Business Impact & Use Cases (30 seconds)

### [VERBAL - Quick summary]

**Use Cases:**
> "This same framework applies across Joon Solutions' client industries:
> - **Retail/E-commerce**: Which customers to send coupons?
> - **Financial Services**: Which accounts to offer premium products?
> - **Healthcare**: Which patients benefit from interventions?
> - **SaaS/Digital**: Which users to target for upgrades?

> The common thread: Stop wasting budget on customers who won't respond, focus on those who will."

---

## PART 6: Closing (30 seconds)

### [VERBAL - Strong finish]

> "To summarize: This Decision Intelligence Studio demonstrates the journey Joon Solutions helps clients achieve - from raw data to causal insights to real-time AI-powered decisions.

> It's not just about dashboards showing what happened. It's about prescriptive systems that tell you WHAT TO DO and provide statistical proof that it works.

> I'm excited about the opportunity to bring these capabilities to Joon Solutions' clients and help them move 'beyond dashboards' into AI-powered decision-making.

> Thank you. I'm happy to dive deeper into any component or answer questions."

---

## 🎯 Anticipated Questions & Answers

### Q1: "How does this differ from regular A/B testing?"
> "Traditional A/B testing tells you if a campaign worked ON AVERAGE. Causal ML with heterogeneous treatment effects tells you WHICH INDIVIDUALS it worked for. This enables personalization at scale."

### Q2: "What if we don't have randomized experiments?"
> "DoWhy handles observational data through backdoor adjustment and other identification strategies. We can estimate causal effects from historical data if we have the right confounders measured."

### Q3: "How does this scale to millions of customers?"
> "The model is trained once (batch) but scoring is real-time via the API - 45ms per customer. For batch scoring, we can process millions through parallel processing on cloud platforms."

### Q4: "What cloud platforms does this work with?"
> "The architecture is cloud-agnostic. The data layer can sit on BigQuery, Snowflake, or Databricks. The ML pipeline can run on Vertex AI, SageMaker, or Airflow. This aligns with Joon's technology-agnostic approach."

### Q5: "How do you handle model drift?"
> "The system includes a Drift Detection module that monitors feature distributions using KS-tests and PSI. Alerts trigger when drift exceeds thresholds, prompting model retraining."

### Q6: "What's the learning curve for business users?"
> "The Streamlit dashboard is designed for self-service analytics - no coding required. Technical teams can use the API and Python libraries directly. This dual-interface approach ensures adoption across skill levels."

---

## 📊 Diagrams to Reference During Demo

| Demo Section | Recommended Diagram |
|--------------|---------------------|
| Problem/Solution Overview | Diagram 10: Business Value Flow |
| Technical Architecture | Diagram 1: Overall System Architecture |
| Pipeline Explanation | Diagram 2: Data Pipeline Flow |
| Causal ML Deep Dive | Diagram 6: Causal Estimation Deep Dive |
| Real-time Processing | Diagram 3: Real-Time Processing Flow |
| API Discussion | Diagram 5: API Request/Response Flow |
| Production Deployment | Diagram 7: Model Serving Architecture |

---

## ⏱️ Time Allocation Summary

| Section | Duration | Cumulative |
|---------|----------|------------|
| 1. The Problem | 1:30 | 1:30 |
| 2. Solution Overview | 1:00 | 2:30 |
| 3. Live Demo | 5:00 | 7:30 |
| 4. Technical Deep Dive | 2:00 | 9:30 |
| 5. Use Cases | 0:30 | 10:00 |
| 6. Closing | 0:30 | 10:30 |

**Buffer:** 30 seconds for transitions/technical issues

---

## 💡 Pro Tips

1. **Start with Impact:** Lead with the $600K waste problem - it's memorable and relevant
2. **Show, Don't Tell:** Let the live dashboard do the talking
3. **Connect to Joon:** Reference their "beyond dashboards" and "AI-powered decisions" messaging
4. **Keep Technical Concise:** 2 minutes max - they can ask for more detail
5. **End with Energy:** Your closing should be confident and forward-looking
6. **Practice Transitions:** Smooth navigation between screens shows polish

**Good luck with your interview! 🚀**
