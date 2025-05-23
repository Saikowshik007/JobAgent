#!/bin/bash

# Script to generate self-signed SSL certificates for IP-based HTTPS access
# Run this script in your project root directory

# Get your public IP automatically (you can also set this manually)
echo "🔍 Detecting your public IP address..."
PUBLIC_IP=$(curl -s https://ipinfo.io/ip)
echo "📍 Detected public IP: $PUBLIC_IP"

# You can override the IP here if needed
# PUBLIC_IP="YOUR_STATIC_IP_HERE"

# Create certs directory
mkdir -p nginx/certs

echo "🔐 Generating SSL certificates for IP: $PUBLIC_IP"

# Generate private key
openssl genrsa -out nginx/certs/server.key 2048

# Generate certificate signing request
openssl req -new -key nginx/certs/server.key -out nginx/certs/server.csr -config <(
cat <<EOF
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
O=YourOrganization
OU=IT Department
CN=$PUBLIC_IP

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
IP.1 = $PUBLIC_IP
IP.2 = 127.0.0.1
IP.3 = 0.0.0.0
DNS.1 = localhost
EOF
)

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -in nginx/certs/server.csr -signkey nginx/certs/server.key -out nginx/certs/server.crt -days 365 -extensions v3_req -extfile <(
cat <<EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
IP.1 = $PUBLIC_IP
IP.2 = 127.0.0.1
IP.3 = 0.0.0.0
DNS.1 = localhost
EOF
)

# Set proper permissions
chmod 600 nginx/certs/server.key
chmod 644 nginx/certs/server.crt

# Clean up CSR file
rm nginx/certs/server.csr

echo "✅ SSL certificates generated successfully!"
echo "📁 Certificates location: nginx/certs/"
echo "🔐 Private key: nginx/certs/server.key"
echo "📜 Certificate: nginx/certs/server.crt"
echo "🌐 Certificate valid for IP: $PUBLIC_IP"
echo ""
echo "🚀 Your API will be accessible at: https://$PUBLIC_IP"
echo ""
echo "⚠️  Important: You'll need to accept the self-signed certificate in your browser"
echo "   Visit https://$PUBLIC_IP and click 'Advanced' -> 'Proceed to site'"
echo ""
echo "📝 Update your React app environment variable:"
echo "   REACT_APP_API_BASE_URL=https://$PUBLIC_IP"