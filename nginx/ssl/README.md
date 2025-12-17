# SSL Certificates Placeholder

This directory should contain SSL certificates for HTTPS:

- `fullchain.pem` - Full certificate chain
- `privkey.pem` - Private key

## Getting SSL Certificates with Let's Encrypt

```bash
# Stop nginx if running
docker compose stop nginx

# Get certificates
sudo certbot certonly --standalone -d ai-scientometer.tou.edu.kz

# Copy certificates to this directory
sudo cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/fullchain.pem .
sudo cp /etc/letsencrypt/live/ai-scientometer.tou.edu.kz/privkey.pem .
sudo chown $USER:$USER *.pem

# Update docker-compose.yml to use nginx.conf instead of nginx-no-ssl.conf
# Then restart nginx
docker compose up -d nginx
```

## For Development/Testing Without SSL

The application can run without SSL using `nginx-no-ssl.conf` (default).
