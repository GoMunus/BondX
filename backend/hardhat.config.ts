import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import * as dotenv from "dotenv";

// Load environment variables from .env file
dotenv.config();

// Get environment variables with fallbacks
const ALCHEMY_URL = process.env.ALCHEMY_URL || "";
const PRIVATE_KEY = process.env.PRIVATE_KEY || "";

// Validate required environment variables
if (!ALCHEMY_URL) {
  throw new Error("ALCHEMY_URL environment variable is required");
}
if (!PRIVATE_KEY) {
  throw new Error("PRIVATE_KEY environment variable is required");
}

const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    // Local development network
    hardhat: {
      chainId: 1337,
    },
    // Polygon Mumbai testnet
    mumbai: {
      url: ALCHEMY_URL,
      accounts: [PRIVATE_KEY],
      chainId: 80001,
      gasPrice: 20000000000, // 20 gwei
    },
    // Polygon mainnet (for production)
    polygon: {
      url: process.env.POLYGON_ALCHEMY_URL || "",
      accounts: process.env.POLYGON_PRIVATE_KEY ? [process.env.POLYGON_PRIVATE_KEY] : [],
      chainId: 137,
      gasPrice: 30000000000, // 30 gwei
    },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  },
  mocha: {
    timeout: 40000,
  },
  ethers: {
    version: "6.8.0",
  },
};

export default config;
