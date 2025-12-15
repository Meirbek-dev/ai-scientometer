#!/bin/bash
# AI Scientometer - Deployment Script for Ubuntu 24.04
# Usage: ./deploy.sh [dev|prod]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AI Scientometer Deployment Script ===${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Installing Docker...${NC}"

    # Install Docker on Ubuntu 24.04
    sudo apt-get update
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

    # Add current user to docker group
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installed successfully!${NC}"
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose plugin not found. Please install docker-compose-plugin${NC}"
    exit 1
fi

# Determine deployment mode
MODE=${1:-prod}
echo -e "${GREEN}Deployment mode: ${MODE}${NC}"

# Create SSL directory for production
if [ "$MODE" = "prod" ]; then
    echo -e "${YELLOW}Creating SSL directory...${NC}"
    mkdir -p nginx/ssl

    # Check if SSL certificates exist
    if [ ! -f "nginx/ssl/fullchain.pem" ] || [ ! -f "nginx/ssl/privkey.pem" ]; then
        echo -e "${YELLOW}SSL certificates not found in nginx/ssl/${NC}"
        echo -e "${YELLOW}Please add your SSL certificates:${NC}"
        echo -e "  - nginx/ssl/fullchain.pem"
        echo -e "  - nginx/ssl/privkey.pem"
        echo ""
        echo -e "${YELLOW}You can use Let's Encrypt to generate free certificates:${NC}"
        echo -e "  sudo certbot certonly --standalone -d ai-scientometer.tou.edu.kz"
        echo -e "  sudo cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/fullchain.pem nginx/ssl/"
        echo -e "  sudo cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/privkey.pem nginx/ssl/"
        echo ""
        read -p "Do you want to continue without SSL? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Stop existing containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
if [ "$MODE" = "dev" ]; then
    docker compose -f docker-compose.dev.yml down 2>/dev/null || true
else
    docker compose -f docker-compose.yml down 2>/dev/null || true
fi

# Build and start containers
echo -e "${GREEN}Building and starting containers...${NC}"
if [ "$MODE" = "dev" ]; then
    docker compose -f docker-compose.dev.yml up -d --build
else
    docker compose -f docker-compose.yml up -d --build
fi

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Check service status
echo -e "${GREEN}Checking service status...${NC}"
if [ "$MODE" = "dev" ]; then
    docker compose -f docker-compose.dev.yml ps
else
    docker compose -f docker-compose.yml ps
fi

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""

if [ "$MODE" = "dev" ]; then
    echo -e "Access the application:"
    echo -e "  - Frontend: ${GREEN}http://localhost:3002${NC}"
    echo -e "  - API: ${GREEN}http://localhost:8000${NC}"
    echo -e "  - API Docs: ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  - MongoDB: ${GREEN}mongodb://localhost:27017${NC}"
else
    echo -e "Access the application:"
    echo -e "  - Website: ${GREEN}https://ai-scientometer.tou.edu.kz${NC}"
    echo -e "  - API Docs: ${GREEN}https://ai-scientometer.tou.edu.kz/docs${NC}"
fi

echo ""
echo -e "Useful commands:"
echo -e "  View logs: docker compose -f docker-compose${MODE == 'dev' ? '.dev' : ''}.yml logs -f"
echo -e "  Stop: docker compose -f docker-compose${MODE == 'dev' ? '.dev' : ''}.yml down"
echo -e "  Restart: docker compose -f docker-compose${MODE == 'dev' ? '.dev' : ''}.yml restart"
