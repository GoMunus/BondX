# BondX Deployment Guide

This guide walks you through deploying the BondX smart contract to Polygon Mumbai testnet.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your values:

```bash
# Required: Alchemy API endpoint for Polygon Mumbai
ALCHEMY_URL="https://polygon-mumbai.g.alchemy.com/v2/YOUR_ALCHEMY_API_KEY"

# Required: Your wallet private key (without 0x prefix)
PRIVATE_KEY="your_private_key_here"
```

**⚠️ Security Note**: Never commit your `.env` file to version control!

### 3. Get Test MATIC

You'll need test MATIC tokens for gas fees:
- **Faucet**: [Polygon Mumbai Faucet](https://faucet.polygon.technology/)
- **Alternative**: [Alchemy Faucet](https://mumbaifaucet.com/)

## 🔧 Deployment Steps

### Step 1: Compile Contracts

```bash
npm run compile
```

**Expected Output**: Contracts compiled successfully with artifacts generated.

### Step 2: Deploy to Mumbai Testnet

```bash
npm run deploy
```

**Expected Output**:
```
🚀 Starting Bond1155 contract deployment...
📝 Deploying Bond1155 contract with base URI: https://bondx-metadata.com/api/token/
✅ Bond1155 contract deployed successfully!
📍 Contract Address: 0x...
🔗 Base URI: https://bondx-metadata.com/api/token/
👤 Owner: 0x...
```

**⚠️ Important**: Save the contract address from the output!

### Step 3: Update Seed Script

Edit `scripts/seed.ts` and replace:
```typescript
const CONTRACT_ADDRESS = "YOUR_DEPLOYED_CONTRACT_ADDRESS_HERE";
```

With your actual contract address:
```typescript
const CONTRACT_ADDRESS = "0x..."; // Your deployed address
```

### Step 4: Seed Initial Bonds

```bash
npm run seed
```

**Expected Output**:
```
🌱 Starting BondX token seeding...
✅ Owner verification passed
📊 Creating bond with parameters:
   🆔 Token ID: 101
   💰 Price: 0.01 ETH
   📦 Supply: 1000 tokens
   🔗 URI: ipfs://sample-uri-or-http-url
✅ Bond created successfully!
```

## 🧪 Testing Your Deployment

### 1. Verify on Polygonscan

1. Go to [Mumbai Polygonscan](https://mumbai.polygonscan.com/)
2. Search for your contract address
3. Verify the contract is deployed and has transactions

### 2. Test Contract Functions

You can test the contract using:
- **Hardhat Console**: Interactive testing
- **Frontend dApp**: Connect MetaMask to Mumbai testnet
- **Scripts**: Create custom test scripts

### 3. Run Tests

```bash
npm run test
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Insufficient funds" | Get test MATIC from faucet |
| "Invalid private key" | Check .env file format |
| "Network error" | Verify Alchemy URL and network |
| "Owner only" | Ensure correct private key |

### Debug Commands

```bash
# Check network connection
npx hardhat console --network mumbai

# View contract state
const contract = await ethers.getContractAt("Bond1155", "CONTRACT_ADDRESS")
await contract.getPrice(101)
await contract.getRemaining(101)
```

## 📊 Post-Deployment

### 1. Contract Verification

Consider verifying your contract on Polygonscan for transparency.

### 2. Frontend Integration

Connect your dApp frontend to the deployed contract.

### 3. Monitoring

Monitor contract activity and gas usage.

## 🚨 Security Checklist

- [ ] Private keys not committed to git
- [ ] Contract deployed to testnet first
- [ ] All functions tested thoroughly
- [ ] Access controls verified
- [ ] Gas limits appropriate

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify environment variables
3. Ensure sufficient test MATIC
4. Check network connectivity

---

**Happy Deploying! 🎉**
