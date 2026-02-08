Here is the comprehensive report on Azure Free Services, structured for your AI agents to ingest and strategize workflows.

### **Azure Free Services Report: 2026 Edition**

**Data Source:** Microsoft Azure Official Pricing & Limits
**Target Audience:** AI Workflow Agents & System Architects

---

### **1. Executive Summary**

Azure's free tier is divided into two distinct buckets. Agents must distinguish between them to prevent unexpected billing after Year 1.

* **12-Months Free:** High-value infrastructure (VMs, SQL, Storage) with strict monthly caps. Expires exactly 365 days after signup.
* **Always Free:** Serverless, DevTools, and lightweight AI services that remain free indefinitely (within limits).

---

### **2. Compute & Containers**

*Strategizing Note: Prioritize "Always Free" serverless options (Functions, Container Apps) for long-running low-intensity tasks. Use "12-Month" VMs only for stateful or legacy workloads.*

| Service | Free Limit | Type | SKU / Tier | Notes |
| --- | --- | --- | --- | --- |
| **Azure Virtual Machines** | **750 Hours / month** (for *each* type) | 12 Months | **B1s** (Intel), **B2pts v2** (ARM), **B2ats v2** (AMD) | *Critical:* "750 hours" allows running **one** instance 24/7. Running two instances consumes the quota in 15 days. |
| **Azure Container Apps** | **180,000 vCPU-seconds** + **2 Million Requests** / month | **Always** | Consumption Plan | Ideal for event-driven workers (like your crawler). |
| **Azure Functions** | **1 Million Requests** + 400,000 GB-s / month | **Always** | Consumption Plan | Best for API endpoints and lightweight triggers. |
| **App Service** | **10 Apps** (Shared) | **Always** | **F1 (Free)** Tier | Very limited CPU (60 mins/day). Good for static sites or dev endpoints, not production. |
| **Container Registry** | **100 GB** Storage + 10 Webhooks | 12 Months | Standard Tier | Essential for storing Docker images. |

---

### **3. Databases & Storage**

*Strategizing Note: Database costs are the #1 cause of billing accidents. The "Always Free" SQL offer is generally safer than the "12-Month" offer because it handles idle time better.*

| Service | Free Limit | Type | SKU / Tier | Notes |
| --- | --- | --- | --- | --- |
| **Azure SQL Database** | **100,000 vCore-seconds** / month | **Always** | Serverless General Purpose | Includes 32GB storage. Pauses automatically when inactive. |
| **Azure SQL Database (Legacy)** | **250 GB** Storage | 12 Months | S0 Standard Tier | *Warning:* Does not pause. Safer to use the Serverless option above. |
| **Cosmos DB** | **1,000 RU/s** + **25 GB** Storage | **Always*** | Free Tier | *Must select "Apply Free Tier Discount"* when creating the *first* account in the subscription. |
| **PostgreSQL Flexible** | **750 Hours** / month | 12 Months | **B1ms** Instance | Includes 32GB storage + 32GB backup. Enough for 1 instance 24/7. |
| **MySQL Flexible** | **750 Hours** / month | 12 Months | **B1ms** Instance | Includes 32GB storage + 32GB backup. |
| **Blob Storage** | **5 GB** LRS | 12 Months | Hot Access / LRS | 20,000 Read/Write operations included. |
| **File Storage** | **5 GB** LRS | 12 Months | Transaction Optimized | Good for shared configs. |
| **Managed Disks** | **64 GB** x 2 (P6 SSD) | 12 Months | Premium SSD | Useful for the OS disk of your Free VM. |

---

### **4. AI & Cognitive Services**

*Strategizing Note: These are high-value "Always Free" APIs. Agents should batch requests to stay within quotas.*

| Service | Free Limit | Type | Notes |
| --- | --- | --- | --- |
| **Language (Text Analytics)** | **5,000 Records** / month | **Always** | Sentiment analysis, Key phrase extraction. |
| **Translator** | **2 Million Characters** / month | **Always** | Text translation. |
| **Speech to Text** | **5 Audio Hours** / month | **Always** | Standard Multichannel Audio. |
| **Text to Speech** | **500,000 Characters** / month | **Always** | Neural voices. |
| **Computer Vision** | **5,000 Transactions** / month | 12 Months | S1, S2, S3 tiers. |
| **Custom Vision** | **10,000 Predictions** / month | **Always** | S0 Tier. Includes 1 training hour. |
| **Content Safety** | **5,000 Records** (Text/Image) | **Always** | Essential for moderating AI inputs/outputs. |
| **OpenAI (Azure)** | *Not Free* | N/A | *Warning:* There is no free tier for Azure OpenAI. |

---

### **5. Web & Networking**

*Strategizing Note: Bandwidth is generous, but Load Balancers are a hidden cost after 12 months.*

| Service | Free Limit | Type | Notes |
| --- | --- | --- | --- |
| **Bandwidth (Outbound)** | **100 GB** / month | **Always** | Global data transfer out. |
| **Static Web Apps** | **100 GB** Bandwidth / month | **Always** | Free Tier. Includes 2 Custom Domains + SSL. |
| **Load Balancer** | **1 Free IP** | 12 Months | Standard Load Balancer. |
| **DevOps** | **5 Users** | **Always** | Unlimited private Git repositories & Boards. |

---

### **6. Strategic "Gotchas" for Agents**

1. **The "Stop" vs. "Deallocate" Trap:**
* **Rule:** Stopping a VM in the OS (Shutdown) **does not** stop the billing clock.
* **Fix:** Agents must execute the Azure API command `Deallocate` to stop the meter.


2. **The 750-Hour Hard Limit:**
* 750 hours  31 days.
* If you run **one** B1s VM, it is free.
* If you run **two** B1s VMs for testing, you burn the month's allowance in 15 days. Days 16-30 will be billed.


3. **OS Disk Cost:**
* While the *Compute* (B1s) is free, the *Disk* (P6 SSD) is only free for 12 months. After year 1, you pay for the disk even if the VM is off.


4. **Region Matters:**
* Free services are available in most regions, but not all. Stick to major hubs (e.g., East US, West Europe, Central India) to ensure B1s/B2pts availability.



**Recommendation:** Configure your **Azure Budget Alerts** immediately to trigger at **$0.01**. This is the only way to know instantly if an agent has accidentally provisioned a paid SKU.