# Cheque OCR Processing System
## Production Deployment for 1 × 4-core CPUs

### Overview
This system processes cheque images (TIFF format) using PaddleOCR, extracting payee names with support for 10+ cheque types and government payee rules. Optimized for 1 × 4-core CPUs, it processes 50,000 cheques in 50-62 minutes.

### Features
✅ 10+ cheque types (Bearer, Order, Crossed, DD, MC, Govt I/II/III, etc.)
✅ Government payee rule enforcement (Category I/II/III)
✅ Title removal (Dr, Adv, M/s, etc.)
✅ & → AND replacement
✅ XXX fallback for null/handwritten
✅ DBF file integration
✅ TIFF format support
✅ 1 × 4-core CPU parallel processing
✅ 800-1000 images/minute throughput

### System Requirements
- **CPU**: 1 × 4-core (4 cores total)
- **RAM**: GB minimum
- **Disk**: 400GB+ NVMe (or process in batches)
- **OS**: Ubuntu 20.04/22.04 LTS (recommended)

### Installation

1. **Clone/Download files**
```bash
git clone <repository>
cd cheque-ocr