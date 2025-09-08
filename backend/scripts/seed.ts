import { ethers } from "hardhat";

/**
 * @dev Seed script for creating initial bond tokens
 * This script calls createBond to mint the first bond series
 * Make sure to update the CONTRACT_ADDRESS with your deployed contract address
 */
async function main() {
  console.log("🌱 Starting BondX token seeding...");
  
  // IMPORTANT: Update this address with your deployed contract address
  const CONTRACT_ADDRESS = "YOUR_DEPLOYED_CONTRACT_ADDRESS_HERE";
  
  if (CONTRACT_ADDRESS === "YOUR_DEPLOYED_CONTRACT_ADDRESS_HERE") {
    console.error("❌ Please update CONTRACT_ADDRESS in the seed script with your deployed contract address");
    console.log("💡 Run 'npm run deploy' first to get the contract address");
    process.exit(1);
  }
  
  try {
    // Get the deployed contract instance
    const Bond1155 = await ethers.getContractFactory("Bond1155");
    const bondContract = Bond1155.attach(CONTRACT_ADDRESS);
    
    console.log(`📋 Contract Address: ${CONTRACT_ADDRESS}`);
    console.log(`👤 Connected as: ${await bondContract.signer.getAddress()}`);
    
    // Check if we're the contract owner
    const owner = await bondContract.owner();
    const signerAddress = await bondContract.signer.getAddress();
    
    if (owner !== signerAddress) {
      console.error("❌ Only the contract owner can create bonds");
      console.log(`👑 Contract Owner: ${owner}`);
      console.log(`🔑 Your Address: ${signerAddress}`);
      process.exit(1);
    }
    
    console.log("✅ Owner verification passed");
    
    // Bond parameters for the first bond series
    const tokenId = 101;
    const priceWei = ethers.parseEther("0.01"); // 0.01 ETH per token
    const supply = 1000; // 1000 tokens total
    const uri = "ipfs://sample-uri-or-http-url";
    
    console.log("\n📊 Creating bond with parameters:");
    console.log(`   🆔 Token ID: ${tokenId}`);
    console.log(`   💰 Price: ${ethers.formatEther(priceWei)} ETH`);
    console.log(`   📦 Supply: ${supply} tokens`);
    console.log(`   🔗 URI: ${uri}`);
    
    // Create the bond
    console.log("\n🚀 Creating bond...");
    const tx = await bondContract.createBond(tokenId, priceWei, supply, uri);
    
    console.log(`📝 Transaction hash: ${tx.hash}`);
    console.log("⏳ Waiting for transaction confirmation...");
    
    // Wait for transaction to be mined
    const receipt = await tx.wait();
    
    console.log("✅ Bond created successfully!");
    console.log(`🔗 Transaction: ${tx.hash}`);
    console.log(`⛽ Gas used: ${receipt.gasUsed.toString()}`);
    
    // Verify the bond was created correctly
    console.log("\n🔍 Verifying bond creation...");
    
    const createdPrice = await bondContract.getPrice(tokenId);
    const createdSupply = await bondContract.getRemaining(tokenId);
    const createdUri = await bondContract.uri(tokenId);
    
    console.log(`✅ Price verification: ${ethers.formatEther(createdPrice)} ETH`);
    console.log(`✅ Supply verification: ${createdSupply} tokens`);
    console.log(`✅ URI verification: ${createdUri}`);
    
    // Check contract balance of the new token
    const contractBalance = await bondContract.balanceOf(CONTRACT_ADDRESS, tokenId);
    console.log(`✅ Contract token balance: ${contractBalance} tokens`);
    
    console.log("\n🎉 Bond seeding completed successfully!");
    console.log("\n📋 Next steps:");
    console.log("1. Test the buy function with a small amount");
    console.log("2. Verify token transfers work correctly");
    console.log("3. Test the withdraw function (owner only)");
    
  } catch (error) {
    console.error("💥 Bond seeding failed:", error);
    
    if (error instanceof Error) {
      console.error("Error details:", error.message);
    }
    
    process.exitCode = 1;
  }
}

// Handle script errors
main().catch((error) => {
  console.error("💥 Script execution failed:", error);
  process.exitCode = 1;
});
