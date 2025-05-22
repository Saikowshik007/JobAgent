#!/bin/bash

# Script to generate self-signed SSL certificates for development
# For production, use Let's Encrypt or a proper CA-signed certificate

# Create nginx/certs directory if it doesn't exist
mkdir -p nginx/certs

# Generate private key
openssl genpkey -algorithm RSA -out nginx/certs/server.key -pkcs8 -pkeyopt rsa_keygen_bits:2048

# Generate certificate signing request
openssl req -new -key nginx/certs/server.key -out nginx/certs/server.csr -subj "/C=US/ST=State/L=City/O=Organization/OU=IT/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -days 365 -in nginx/certs/server.csr -signkey nginx/certs/server.key -out nginx/certs/server.crt

# Set appropriate permissions
chmod 600 nginx/certs/server.key
chmod 644 nginx/certs/server.crt

# Clean up CSR file
rm nginx/certs/server.csr

echo "SSL certificates generated successfully!"
echo "Private key: nginx/certs/server.key"
echo "Certificate: nginx/certs/server.crt"
echo ""
echo "⚠️  WARNING: These are self-signed certificates for development only!"
echo "   For production, use proper CA-signed certificates or Let's Encrypt."