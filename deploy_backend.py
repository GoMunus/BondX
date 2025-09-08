#!/usr/bin/env python3
"""
Quick Backend Deployment Script for BondX
Deploy your backend to make it accessible to your frontend at https://ui-bond-x.vercel.app/
"""

import os
import sys

def main():
    print("🚀 BondX Backend Deployment Options:")
    print("\n1. **RAILWAY (Recommended - Free)**")
    print("   - Visit: https://railway.app")
    print("   - Connect your GitHub repo")
    print("   - Deploy with one click")
    print("   - Your API will be at: https://your-app.railway.app")
    
    print("\n2. **RENDER (Alternative - Free)**")
    print("   - Visit: https://render.com")
    print("   - Connect GitHub")
    print("   - Deploy as Web Service")
    
    print("\n3. **HEROKU (Paid)**")
    print("   - Visit: https://heroku.com")
    print("   - Create new app")
    print("   - Deploy via Git")
    
    print("\n4. **LOCAL TESTING (For now)**")
    print("   - Run: python start_backend.py")
    print("   - Use ngrok to expose: ngrok http 8000")
    print("   - Get public URL like: https://abc123.ngrok.io")
    
    print("\n" + "="*50)
    print("📋 IMPORTANT: After deployment, update your frontend's API URL")
    print("   Replace dummy data calls with your deployed backend URL")

if __name__ == "__main__":
    main()
