# AI Scientometer - Docker Deployment Guide

## Prerequisites

- Ubuntu 24.04 server with SSH access
- Domain: `ai-scientometer.tou.edu.kz` pointing to server IP `192.168.12.35`
- Docker and Docker Compose installed

## Quick Start

### 1. Connect to Server

```bash
ssh user@192.168.12.35
```

### 2. Clone the Repository

```bash
git clone <your-repo-url> ai-scientometer
cd ai-scientometer
```

### 3. Install Docker (if not installed)

```bash
# Update packages
sudo apt-get update

# Install Docker
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 4. Deploy Without SSL (Initial Setup)

```bash
# Create SSL directory (empty for now)
mkdir -p nginx/ssl nginx/certbot

# Start the application
docker compose up -d --build

# Check status
docker compose ps

# View logs
docker compose logs -f
```

The application will be available at:
- Frontend: `http://192.168.12.35` or `http://ai-scientometer.tou.edu.kz`
- API Docs: `http://ai-scientometer.tou.edu.kz/docs`

### 5. Enable SSL with Let's Encrypt

```bash
# Install certbot
sudo apt-get install -y certbot

# Stop nginx temporarily
docker compose stop nginx

# Get SSL certificate
sudo certbot certonly --standalone -d ai-scientometer.tou.edu.kz

# Copy certificates
sudo cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/privkey.pem nginx/ssl/
sudo chown $USER:$USER nginx/ssl/*.pem

# Update docker-compose.yml to use SSL config
# Change this line in docker-compose.yml:
#   - ./nginx/nginx-no-ssl.conf:/etc/nginx/nginx.conf:ro
# To:
#   - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro

# Restart nginx
docker compose up -d nginx
```

### 6. Auto-renewal of SSL Certificates

```bash
# Create renewal script
cat > /home/$USER/renew-ssl.sh << 'EOF'
#!/bin/bash
cd /home/$USER/ai-scientometer
docker compose stop nginx
certbot renew --quiet
cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/fullchain.pem nginx/ssl/
cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/privkey.pem nginx/ssl/
docker compose start nginx
EOF

chmod +x /home/$USER/renew-ssl.sh

# Add to crontab (runs twice daily)
(crontab -l 2>/dev/null; echo "0 0,12 * * * /home/$USER/renew-ssl.sh") | crontab -
```

## Useful Commands

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f web
docker compose logs -f nginx
docker compose logs -f mongodb
```

### Restart Services

```bash
# All services
docker compose restart

# Specific service
docker compose restart api
```

### Update Application

```bash
git pull
docker compose up -d --build
```

### Stop Application

```bash
docker compose down
```

### Clean Up

```bash
# Remove containers and volumes (WARNING: deletes data!)
docker compose down -v

# Remove unused images
docker image prune -f
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Internet                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Nginx (Port 3002/443)                       │
│                  scientometer-nginx                          │
└─────────────────────────────────────────────────────────────┘
                    │                    │
          /api/*    │                    │    /*
                    ▼                    ▼
┌───────────────────────┐    ┌───────────────────────┐
│   FastAPI Backend     │    │   React Frontend      │
│   scientometer-api    │    │   scientometer-web    │
│   (Port 8000)         │    │   (Port 80)           │
└───────────────────────┘    └───────────────────────┘
                    │
                    ▼
┌───────────────────────┐
│      MongoDB          │
│  scientometer-mongodb │
│   (Port 27017)        │
└───────────────────────┘
```

## Volumes

| Volume         | Description                 |
| -------------- | --------------------------- |
| `mongodb_data` | MongoDB database files      |
| `api_datasets` | AI models and training data |

## Environment Variables

| Variable        | Default                                  | Description               |
| --------------- | ---------------------------------------- | ------------------------- |
| `MONGODB_URL`   | `mongodb://mongodb:27017`                | MongoDB connection string |
| `DATABASE_NAME` | `scientometer`                           | Database name             |
| `VITE_API_URL`  | `https://ai-scientometer.tou.edu.kz/api` | API URL for frontend      |

## Troubleshooting

### API not starting

```bash
# Check API logs
docker compose logs api

# Check if MongoDB is healthy
docker compose ps mongodb
```

### Frontend showing connection error

1. Check if API is running: `curl http://localhost:8000/health`
2. Check nginx configuration
3. Verify CORS settings

### SSL certificate issues

```bash
# Check certificate validity
openssl x509 -in nginx/ssl/fullchain.pem -text -noout

# Regenerate certificates
sudo certbot certonly --standalone -d ai-scientometer.tou.edu.kz --force-renewal
```

### High memory usage by AI model

The AI model (sentence-transformers) loads into memory on startup. Ensure the server has at least 4GB RAM.

## Support

For issues, please create a GitHub issue in the repository.
