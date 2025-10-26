#!/bin/bash
# Setup script for Playwright browser installation

echo "======================================"
echo "Playwright Setup for JobAgent"
echo "======================================"
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install Python and pip first."
    exit 1
fi

# Install playwright package
echo "📦 Installing playwright package..."
pip install playwright>=1.40.0

# Check installation
if [ $? -ne 0 ]; then
    echo "❌ Failed to install playwright package"
    exit 1
fi

echo "✓ Playwright package installed"
echo ""

# Install chromium browser
echo "📥 Installing Chromium browser..."
playwright install chromium

# Check installation
if [ $? -ne 0 ]; then
    echo "❌ Failed to install Chromium browser"
    echo ""
    echo "Try running with system dependencies:"
    echo "  playwright install --with-deps chromium"
    exit 1
fi

echo "✓ Chromium browser installed"
echo ""

# Install system dependencies (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Installing system dependencies (Linux)..."
    playwright install-deps chromium

    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Some system dependencies might be missing"
        echo "   This is usually OK, but if Playwright fails, run:"
        echo "   sudo playwright install-deps chromium"
    else
        echo "✓ System dependencies installed"
    fi
fi

echo ""
echo "======================================"
echo "✅ Playwright setup complete!"
echo "======================================"
echo ""
echo "Installed browsers:"
playwright --version
echo ""
echo "Test with:"
echo "  python -c \"from playwright.sync_api import sync_playwright; p = sync_playwright().start(); print('✓ Playwright working!'); p.stop()\""
