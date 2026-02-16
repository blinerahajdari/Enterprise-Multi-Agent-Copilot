# Supply Chain Knowledge Base (Synthetic Documents)

This folder contains synthetic supply chain documents used for retrieval and grounding in the Enterprise Multi-Agent Copilot.

All documents are fictional and created solely for academic purposes.

---

## Document List

### 1. Disruption Response Playbook
Describes the standard operating procedure when a supply disruption occurs, including:
- War-room activation
- Expedite vs alternate sourcing decisions
- Post-incident review process  
Reference: SC_ActionPlaybook_DisruptionResponse.txt

---

### 2. Contract SLA Clauses
Defines contractual performance commitments including:
- OTIF thresholds
- Volume reallocation rights
- Lead time change rules
- Capacity reservation clauses  
Reference: SC_ContractClauses_SLA.txt

---

### 3. Dual Sourcing Cost Model
Compares single-source vs dual-source scenarios for SKU-A and SKU-B, including:
- Unit cost assumptions
- Premium freight exposure
- Break-even threshold  
Reference: SC_CostModel_DualSourcing.txt

---

### 4. Demand Forecast Notes (Q1 2026)
Provides forecast assumptions and error margins for priority SKUs.
Includes seasonality adjustments and planning notes.  
Reference: SC_DemandForecast_Notes.txt

---

### 5. Incident Report – Port Delay (Nov 2025)
Describes a real disruption scenario including:
- Lead time impact
- Premium freight cost
- Temporary mitigation actions  
Reference: SC_IncidentReport_PortDelay_Nov2025.txt

---

### 6. Inventory Policy – Buffer Stock
Defines:
- Target service levels
- Safety stock calculation logic
- Temporary risk buffers  
Reference: SC_InventoryPolicy_BufferStock.txt

---

### 7. Supply Chain KPI Definitions
Defines core KPIs:
- OTIF
- Fill Rate
- Expedite Rate
- Reporting cadence  
Reference: SC_KPI_Definitions_OTIF_FillRate.txt

---

### 8. Lead Time Analysis
Historical analysis of contracted vs actual lead times for suppliers:
- NSC
- DP
- CM  
Reference: SC_LeadTime_Analysis_Suppliers.txt

---

### 9. Supply Chain Risk Register – 2025
Identifies key operational risks including:
- Port congestion
- Single-source dependency
- Capacity constraints  
Reference: SC_RiskRegister_2025.txt

---

### 10. Supplier Scorecard – Q4 2025
Supplier performance metrics:
- OTIF
- Lead time variability
- Capacity constraints
- Premium freight exposure  
Reference: SC_SupplierScorecard_Q4_2025.txt

---

## Purpose

These documents serve as the retrieval foundation for the multi-agent system.  
The Research Agent retrieves evidence from these files, and the Verifier Agent ensures all claims are supported by citations.

If a requested fact does not exist in these documents, the system must respond:

Not found in sources.