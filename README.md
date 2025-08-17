# BondX Backend - AI-Powered Fractional Bond Marketplace

A sophisticated FastAPI backend designed specifically for the Indian debt capital market, featuring advanced bond pricing engines, comprehensive financial mathematics, and machine learning infrastructure.

## 🏗️ Architecture Overview

### Core Components
- **FastAPI Backend**: High-performance async API with comprehensive documentation
- **PostgreSQL Database**: Sophisticated schema for Indian bond markets
- **Redis Cache**: High-performance caching and session management
- **Celery**: Background task processing for data ingestion and analytics
- **Advanced Mathematics**: Comprehensive fixed-income analytics and yield calculations

### Key Features
- Multi-day count convention support (30/360, ACT/ACT, ACT/365, 30/365)
- Advanced yield-to-maturity calculations with Newton-Raphson iteration
- Duration and convexity analytics including modified, Macaulay, and effective duration
- Option-adjusted spread (OAS) calculations for callable/putable bonds
- Real-time market data integration from NSE, BSE, and RBI
- Credit rating tracking from CRISIL, ICRA, CARE, Moody's, and Fitch
- Comprehensive regulatory compliance and audit trails

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- Poetry (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bondx-backend
   ```

2. **Install dependencies**
   ```bash
   poetry install
   # or
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize database**
   ```bash
   poetry run alembic upgrade head
   ```

5. **Run the application**
   ```bash
   poetry run uvicorn bondx.main:app --reload
   ```

## 📊 Database Schema

### Core Tables
- **instruments**: Complete bond metadata with ISIN codes, issuer hierarchies, coupon structures
- **market_quotes**: Real-time pricing with clean/dirty conventions and yield calculations
- **yield_curves**: Multiple curve types (G-sec, SDL, corporate) with term structure data
- **corporate_actions**: Splits, mergers, rating changes, and regulatory announcements
- **macro_indicators**: Repo rates, CPI, industrial production data

### Advanced Features
- Embedded options tracking (call/put provisions)
- Credit rating history from multiple agencies
- Sector classifications aligned with RBI guidelines
- Regulatory status and compliance tracking

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret for authentication
- `API_KEY`: External API keys for data sources
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Database Configuration
- Connection pooling with asyncpg
- Alembic migrations for schema management
- Comprehensive indexing for performance
- Data partitioning for large datasets

## 📈 API Endpoints

### Core Bond Operations
- `GET /api/v1/bonds`: List bonds with advanced filtering
- `GET /api/v1/bonds/{isin}`: Detailed bond information
- `POST /api/v1/bonds/{isin}/price`: Calculate bond pricing
- `GET /api/v1/bonds/{isin}/yield`: Yield-to-maturity calculations
- `GET /api/v1/bonds/{isin}/duration`: Duration and convexity analytics

### Market Data
- `GET /api/v1/market/quotes`: Real-time market quotes
- `GET /api/v1/market/yield-curves`: Yield curve data
- `GET /api/v1/market/ratings`: Credit rating updates
- `WebSocket /ws/market/streams`: Real-time price feeds

### Analytics
- `POST /api/v1/analytics/portfolio`: Portfolio analytics
- `GET /api/v1/analytics/correlation`: Correlation analysis
- `GET /api/v1/analytics/volatility`: Volatility calculations
- `POST /api/v1/analytics/regime-detection`: Regime detection

## 🧮 Financial Mathematics

### Bond Pricing Engine
- **Day Count Conventions**: 30/360, ACT/ACT, ACT/365, 30/365
- **Accrued Interest**: Complex scenarios with irregular periods
- **Yield Calculations**: Newton-Raphson with intelligent initial guesses
- **Duration Analytics**: Modified, Macaulay, effective, and key rate durations
- **Convexity**: Price sensitivity to yield changes

### Advanced Features
- **Option-Adjusted Spread**: Monte Carlo and binomial tree implementations
- **Cash Flow Projections**: Complex coupon structures and embedded options
- **Yield Curve Modeling**: Bootstrapping, par rates, forward rates
- **Risk Metrics**: VaR, stress testing, scenario analysis

## 🔍 Data Integration

### Market Data Sources
- **NSE & BSE**: Debt segment data with anti-bot handling
- **RBI**: Government securities, monetary policy, macro indicators
- **Rating Agencies**: CRISIL, ICRA, CARE, Moody's, Fitch
- **News Sources**: Economic Times, Business Standard, LiveMint

### Data Quality
- Automated validation rules and outlier detection
- Data freshness tracking and versioning
- Comprehensive error handling with circuit breakers
- Audit trails for regulatory compliance

## 🧪 Testing

### Test Suite
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=bondx

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/performance/
```

### Load Testing
```bash
# Run load tests with Locust
poetry run locust -f tests/load/locustfile.py
```

## 📚 Documentation

### API Documentation
- Interactive API docs at `/docs` (Swagger UI)
- Alternative docs at `/redoc` (ReDoc)
- OpenAPI schema at `/openapi.json`

### Code Documentation
- Comprehensive docstrings and type hints
- Sphinx documentation generation
- Architecture decision records (ADRs)

## 🔒 Security & Compliance

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- API key management with rotation
- Rate limiting and abuse prevention

### Regulatory Compliance
- SEBI and RBI compliance features
- Comprehensive audit trails
- Data retention policies
- Secure data handling and encryption

## 🚀 Deployment

### Docker
```bash
docker-compose up -d
```

### Production Considerations
- Horizontal scaling with load balancers
- Database clustering and replication
- Redis cluster for high availability
- Monitoring and alerting with Prometheus/Grafana
- Log aggregation with ELK stack

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with comprehensive tests
4. Ensure code quality with pre-commit hooks
5. Submit a pull request

### Code Quality
- Black for code formatting
- Flake8 for linting
- MyPy for type checking
- Pre-commit hooks for automated checks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation and examples

## 🔮 Roadmap

### Phase 1: Data Foundation ✅
- Database architecture and schema design
- Advanced bond mathematics implementation
- Market data integration framework
- Statistical foundation for ML

### Phase 2: AI/ML Integration (Coming Soon)
- Machine learning model development
- Predictive analytics and forecasting
- Natural language processing for news
- Portfolio optimization algorithms

### Phase 3: Advanced Features (Future)
- Real-time trading capabilities
- Advanced risk management
- Regulatory reporting automation
- Mobile application backend
