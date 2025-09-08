# BondX Blockchain Backend

A fractional corporate bond trading platform built on Polygon using ERC-1155 smart contracts.

## 🚀 Features

- **ERC-1155 Multi-Token Standard**: Support for multiple bond series with unique token IDs
- **Fractional Ownership**: Buy and sell fractional portions of corporate bonds
- **Owner Controls**: Contract owner can create new bonds and withdraw funds
- **Automated Supply Management**: Tracks remaining supply for each bond series
- **Polygon Network**: Deployed on Polygon Mumbai testnet for cost-effective transactions

## 📋 Prerequisites

- Node.js 18+ and npm
- MetaMask or similar wallet
- Alchemy account for Polygon Mumbai testnet
- Test MATIC tokens for gas fees

## 🛠️ Installation

1. **Clone and navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your actual values:
   - `ALCHEMY_URL`: Your Alchemy API endpoint for Polygon Mumbai
   - `PRIVATE_KEY`: Your wallet's private key (without 0x prefix)

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ALCHEMY_URL` | Alchemy API endpoint for Polygon Mumbai | `https://polygon-mumbai.g.alchemy.com/v2/YOUR_KEY` |
| `PRIVATE_KEY` | Your wallet private key | `abc123...` (without 0x) |

### Network Configuration

The project is configured for:
- **Local Development**: Hardhat local network
- **Testnet**: Polygon Mumbai (chain ID: 80001)
- **Mainnet**: Polygon (chain ID: 137) - optional

## 📜 Smart Contract

### Bond1155.sol

The main smart contract that implements:
- Bond creation and management
- Token sales and transfers
- Supply tracking
- Fund withdrawal

#### Key Functions

- `createBond()`: Create new bond series (owner only)
- `buy()`: Purchase bond tokens
- `withdraw()`: Withdraw collected funds (owner only)
- `getPrice()`: Get token price
- `getRemaining()`: Get remaining supply

## 🚀 Deployment

### 1. Compile Contracts

```bash
npm run compile
```

### 2. Deploy to Mumbai Testnet

```bash
npm run deploy
```

**Important**: Save the deployed contract address from the output.

### 3. Seed Initial Bonds

Update the `CONTRACT_ADDRESS` in `scripts/seed.ts`, then run:

```bash
npm run seed
```

This creates:
- Token ID: 101
- Price: 0.01 MATIC per token
- Supply: 1000 tokens
- URI: Sample metadata URI

## 🧪 Testing

### Local Testing

```bash
npm run test
```

### Testnet Testing

1. **Buy tokens:**
   ```typescript
   // Example: Buy 10 tokens of bond 101
   const amount = 10;
   const price = await contract.getPrice(101);
   const totalCost = price * amount;
   
   await contract.buy(101, amount, { value: totalCost });
   ```

2. **Check balances:**
   ```typescript
   const balance = await contract.balanceOf(userAddress, 101);
   const remaining = await contract.getRemaining(101);
   ```

## 📊 Contract Interaction Examples

### Creating a New Bond

```typescript
const tokenId = 102;
const price = ethers.parseEther("0.02"); // 0.02 MATIC
const supply = 500;
const uri = "ipfs://new-bond-metadata";

await bondContract.createBond(tokenId, price, supply, uri);
```

### Purchasing Bonds

```typescript
const tokenId = 101;
const amount = 5;
const price = await bondContract.getPrice(tokenId);
const totalCost = price * amount;

await bondContract.buy(tokenId, amount, { value: totalCost });
```

### Withdrawing Funds

```typescript
const recipient = "0x..."; // Recipient address
await bondContract.withdraw(recipient);
```

## 🔍 Verification

### Contract Verification

After deployment, verify your contract on Polygonscan:
1. Go to [Mumbai Polygonscan](https://mumbai.polygonscan.com/)
2. Search for your contract address
3. Click "Contract" tab
4. Click "Verify and Publish"

### Function Verification

Test all contract functions:
- ✅ Bond creation
- ✅ Token purchases
- ✅ Supply tracking
- ✅ Fund withdrawal
- ✅ Owner controls

## 🚨 Security Considerations

- **Private Keys**: Never commit private keys to version control
- **Test Networks**: Always test on testnets before mainnet
- **Access Control**: Only the contract owner can create bonds and withdraw funds
- **Supply Validation**: Contract prevents overselling and invalid transfers

## 📁 Project Structure

```
backend/
├── contracts/
│   └── Bond1155.sol          # Main smart contract
├── scripts/
│   ├── deploy.ts             # Deployment script
│   └── seed.ts               # Initial bond creation
├── test/                     # Test files
├── hardhat.config.ts         # Hardhat configuration
├── package.json              # Dependencies and scripts
├── tsconfig.json             # TypeScript configuration
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## 🆘 Troubleshooting

### Common Issues

1. **"Insufficient funds"**: Ensure your wallet has enough MATIC for gas
2. **"Only owner" errors**: Verify you're using the correct private key
3. **"Invalid network"**: Check your MetaMask network settings
4. **"Contract not found"**: Verify the contract address in seed script

### Getting Help

- Check transaction logs on Polygonscan
- Verify environment variables are set correctly
- Ensure you're connected to the right network
- Check gas limits and prices

## 🚀 Next Steps

After successful deployment:

1. **Frontend Integration**: Connect your dApp frontend to the contract
2. **Metadata Management**: Set up IPFS or HTTP endpoints for token metadata
3. **Additional Features**: Implement bond maturity, interest payments, etc.
4. **Mainnet Deployment**: Deploy to Polygon mainnet for production use

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

This is a hackathon MVP. For production use, consider:
- Additional security audits
- Gas optimization
- Enhanced error handling
- Comprehensive testing suite
- Multi-signature controls

---

**Built with ❤️ for the BondX fractional bond trading platform**
