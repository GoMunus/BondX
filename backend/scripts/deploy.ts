import { ethers } from "hardhat";

/**
 * @dev Deployment script for Bond1155 contract
 * This script deploys the contract and logs the deployed address
 */
async function main() {
  console.log("🚀 Starting Bond1155 contract deployment...");
  
  // Get the contract factory
  const Bond1155 = await ethers.getContractFactory("Bond1155");
  
  // Base URI for token metadata (can be updated later)
  const baseURI = "https://bondx-metadata.com/api/token/";
  
  console.log(`📝 Deploying Bond1155 contract with base URI: ${baseURI}`);
  
  // Deploy the contract
  const bondContract = await Bond1155.deploy(baseURI);
  
  // Wait for deployment to complete
  await bondContract.waitForDeployment();
  
  // Get the deployed contract address
  const contractAddress = await bondContract.getAddress();
  
  console.log("✅ Bond1155 contract deployed successfully!");
  console.log(`📍 Contract Address: ${contractAddress}`);
  console.log(`🔗 Base URI: ${baseURI}`);
  console.log(`👤 Owner: ${await bondContract.owner()}`);
  
  // Verify deployment by checking contract state
  console.log("\n🔍 Verifying deployment...");
  
  try {
    const owner = await bondContract.owner();
    console.log(`✅ Owner verification: ${owner}`);
    
    // Check if contract has the correct interface
    const supportsInterface = await bondContract.supportsInterface("0xd9b67a26");
    console.log(`✅ ERC-1155 interface support: ${supportsInterface}`);
    
    console.log("\n🎉 Contract deployment and verification completed successfully!");
    console.log("\n📋 Next steps:");
    console.log("1. Save the contract address for future reference");
    console.log("2. Run 'npm run seed' to create initial bond tokens");
    console.log("3. Test the contract functions on Mumbai testnet");
    
  } catch (error) {
    console.error("❌ Verification failed:", error);
  }
}

// Handle deployment errors
main().catch((error) => {
  console.error("💥 Deployment failed:", error);
  process.exitCode = 1;
});
