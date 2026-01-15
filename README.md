# Monitor/Drift Agent

A comprehensive monitoring and anomaly detection agent for system metrics and cloud usage, with automated alerting and optional self-healing capabilities.

## Overview

The Monitor/Drift Agent is designed to:
- Collect system metrics (CPU, memory, disk, network) and cloud usage metrics
- Detect anomalies using threshold-based and machine learning methods
- Enforce policies for monitoring and alerting
- Send alerts through multiple channels (email, Slack, SMS)
- Optionally perform self-healing actions (auto-scaling, shutdown)

## Project Structure

```
monitor_drift_agent/
├── api/                           # API endpoints for agent communication
│   ├── __init__.py
│   ├── metrics.py                 # API to collect system metrics
│   ├── alerts.py                 # API to trigger and manage alerts
│   └── policies.py               # API to manage policies
├── data_collection/               # Modules related to data collection
│   ├── __init__.py
│   ├── system_metrics.py         # Logic for collecting system metrics
│   ├── cloud_metrics.py          # Logic for collecting cloud usage metrics
│   └── database.py               # Database interactions (PostgreSQL)
├── anomaly_detection/             # Anomaly detection logic
│   ├── __init__.py
│   ├── threshold_detection.py   # Threshold-based anomaly detection
│   ├── machine_learning.py       # ML-based detection (Isolation Forest, Autoencoders)
│   └── alert_trigger.py          # Logic for triggering alerts based on anomalies
├── policy_management/             # Policy enforcement logic
│   ├── __init__.py
│   ├── policy_definition.py      # Logic for defining and storing policies
│   └── policy_enforcement.py     # Logic for enforcing policies
├── alerting/                      # Modules related to alerts and notifications
│   ├── __init__.py
│   ├── alert_system.py           # Alerting system (email, Slack, SMS)
│   ├── alert_logging.py          # Logic for logging alerts in database
│   └── alert_history.py          # Logic for retrieving alert history
├── self_healing/                  # Self-healing logic (Optional)
│   ├── __init__.py
│   └── auto_scaling.py            # Logic for scaling down resources
├── tests/                         # Unit, Integration, and End-to-End Tests
│   ├── __init__.py
│   ├── test_data_collection.py
│   ├── test_anomaly_detection.py
│   ├── test_alerting.py
│   ├── test_policy_management.py
│   ├── test_self_healing.py
│   ├── test_integration.py
│   └── test_performance.py
├── config/                        # Configuration files
│   ├── __init__.py
│   ├── settings.py               # Configuration settings
│   └── logging_config.py         # Logging configuration
├── utils/                         # Utility functions and classes
│   ├── __init__.py
│   └── logger.py                 # Custom logging utility
├── docker/                        # Docker setup for containerization
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
└── README.md                      # This file
```

## Features

### Data Collection
- **System Metrics**: Collect CPU, memory, disk, and network statistics
- **Cloud Metrics**: Support for AWS, Azure, and GCP cost and resource metrics
- **Database Integration**: PostgreSQL for persistent storage

### Anomaly Detection
- **Threshold-Based**: Configurable threshold rules for metric monitoring
- **Machine Learning**: Optional ML-based detection using Isolation Forest and Autoencoders
- **Alert Triggering**: Automatic alert generation for detected anomalies

### Policy Management
- **Policy Definition**: Create and manage monitoring policies
- **Policy Enforcement**: Automatic policy evaluation and action execution
- **Flexible Rules**: Support for complex conditions and actions

### Alerting
- **Multi-Channel**: Email, Slack, and SMS notifications
- **Alert Logging**: Persistent storage of all alerts
- **Alert History**: Query and analyze historical alerts

### Self-Healing (Optional)
- **Auto-Scaling**: Automatic resource scaling based on cost spikes
- **Service Shutdown**: Graceful shutdown of services when needed

## Prerequisites

- Python 3.11 or higher
- PostgreSQL 15 or higher
- Docker and Docker Compose (for containerized deployment)
- Cloud provider credentials (if using cloud metrics collection)

## Installation

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd monitor_drift_agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # TODO: Create requirements.txt with project dependencies
   # pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy example environment file
   # cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up database**
   ```bash
   # Create PostgreSQL database
   createdb monitor_drift
   # Run migrations (when implemented)
   # python -m alembic upgrade head
   ```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **View logs**
   ```bash
   docker-compose logs -f agent
   ```

3. **Stop services**
   ```bash
   docker-compose down
   ```

## Configuration

Configuration is managed through environment variables. Key settings include:

### Database Configuration
- `DATABASE_URL`: PostgreSQL connection string

### Metric Collection
- `METRIC_COLLECTION_INTERVAL`: Collection interval in seconds (default: 60)

### Thresholds
- `DEFAULT_CPU_THRESHOLD`: Default CPU usage threshold (default: 80.0)
- `DEFAULT_MEMORY_THRESHOLD`: Default memory usage threshold (default: 80.0)
- `DEFAULT_DISK_THRESHOLD`: Default disk usage threshold (default: 80.0)

### Cloud Provider Credentials
- AWS: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- Azure: `AZURE_SUBSCRIPTION_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`
- GCP: `GCP_PROJECT_ID`, `GCP_CREDENTIALS_PATH`

### Alert Channels
- Email: `EMAIL_SMTP_SERVER`, `EMAIL_SMTP_PORT`, `EMAIL_USERNAME`, `EMAIL_PASSWORD`
- Slack: `SLACK_WEBHOOK_URL`
- SMS: `SMS_API_KEY`, `SMS_API_SECRET`, `SMS_FROM_NUMBER`

### Feature Flags
- `ANOMALY_DETECTION_ENABLED`: Enable/disable anomaly detection (default: true)
- `ML_DETECTION_ENABLED`: Enable/disable ML-based detection (default: false)
- `SELF_HEALING_ENABLED`: Enable/disable self-healing (default: false)

## Usage

### Running the Agent

```bash
# Local development
python -m api

# Docker
docker-compose up agent
```

### API Endpoints

The agent provides REST API endpoints for:
- **Metrics**: `GET /api/metrics`, `POST /api/metrics/collect`
- **Alerts**: `GET /api/alerts`, `POST /api/alerts`, `PUT /api/alerts/{id}`
- **Policies**: `GET /api/policies`, `POST /api/policies`, `PUT /api/policies/{id}`

### Creating Policies

```python
from policy_management.policy_definition import create_policy_manager

manager = create_policy_manager()
policy = manager.create_policy(
    name="High CPU Alert",
    description="Alert when CPU usage exceeds 80%",
    rules=[{
        "condition": {
            "type": "metric_threshold",
            "metric_name": "cpu_percent",
            "operator": "gt",
            "threshold": 80.0
        },
        "action": {
            "type": "alert",
            "severity": "high",
            "message": "CPU usage is above threshold"
        },
        "priority": 1
    }]
)
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_data_collection.py

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Development

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement feature following the existing module structure
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Specify your license here]

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: [Your contact information]

## Roadmap

- [ ] Implement full metric collection logic
- [ ] Complete ML-based anomaly detection
- [ ] Add more cloud provider integrations
- [ ] Implement database migrations
- [ ] Add API documentation (OpenAPI/Swagger)
- [ ] Create web dashboard for monitoring
- [ ] Add more notification channels
- [ ] Implement advanced self-healing strategies

## Notes

This is a skeleton implementation. All modules contain TODO comments indicating where full implementation is needed. The structure is designed to be modular and scalable, allowing for independent development of each component.
#   M o n i t o r - D r i f t - A g e n t  
 