#!/bin/bash

# Advanced script to generate self-signed SSL certificates with SAN for development
# For production, use Let's Encrypt or a proper CA-signed certificate

# Create nginx/certs directory if it doesn't exist
mkdir -p nginx/certs

# Create OpenSSL configuration file with SAN
cat > nginx/certs/server.conf <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=US
ST=State
L=City
O=Organization
OU=IT Department
CN=localhost

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
DNS.3 = 127.0.0.1
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate private key
echo "Generating private key..."
openssl genrsa -out nginx/certs/server.key 2048

# Generate certificate signing request with SAN
echo "Generating certificate signing request..."
openssl req -new -key nginx/certs/server.key -out nginx/certs/server.csr -config nginx/certs/server.conf

# Generate self-signed certificate (valid for 365 days) with SAN
echo "Generating self-signed certificate..."
openssl x509 -req -days 365 -in nginx/certs/server.csr -signkey nginx/certs/server.key -out nginx/certs/server.crt -extensions v3_req -extfile nginx/certs/server.conf

# Set appropriate permissions
chmod 600 nginx/certs/server.key
chmod 644 nginx/certs/server.crt

# Clean up temporary files
rm nginx/certs/server.csr nginx/certs/server.conf

echo "✅ SSL certificates generated successfully!"
echo "📁 Private key: nginx/certs/server.key"
echo "📁 Certificate: nginx/certs/server.crt"
echo ""
echo "📋 Certificate details:"
openssl x509 -in nginx/certs/server.crt -text -noout | grep -A 1 "Subject:"
openssl x509 -in nginx/certs/server.crt -text -noout | grep -A 5 "Subject Alternative Name"
echo ""
echo "⚠️  WARNING: These are self-signed certificates for development only!"
echo "   For production, use proper CA-signed certificates or Let's Encrypt."
echo ""
echo "🚀 You can now start your services with:"
echo "   docker-compose -f docker-compose-nginx.yml up -d"