import { expect } from "chai";
import { ethers } from "hardhat";
import { Bond1155 } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("Bond1155", function () {
  let bondContract: Bond1155;
  let owner: SignerWithAddress;
  let buyer1: SignerWithAddress;
  let buyer2: SignerWithAddress;
  let baseURI: string;

  beforeEach(async function () {
    // Get signers
    [owner, buyer1, buyer2] = await ethers.getSigners();
    
    // Base URI for testing
    baseURI = "https://bondx-metadata.com/api/token/";
    
    // Deploy contract
    const Bond1155 = await ethers.getContractFactory("Bond1155");
    bondContract = await Bond1155.deploy(baseURI);
  });

  describe("Deployment", function () {
    it("Should set the correct owner", async function () {
      expect(await bondContract.owner()).to.equal(owner.address);
    });

    it("Should set the correct base URI", async function () {
      expect(await bondContract.uri(0)).to.equal(baseURI);
    });
  });

  describe("Bond Creation", function () {
    const tokenId = 101;
    const price = ethers.parseEther("0.01");
    const supply = 1000;
    const uri = "ipfs://test-bond-metadata";

    it("Should allow owner to create a bond", async function () {
      await expect(bondContract.createBond(tokenId, price, supply, uri))
        .to.emit(bondContract, "BondCreated")
        .withArgs(tokenId, price, supply, uri);
      
      expect(await bondContract.getPrice(tokenId)).to.equal(price);
      expect(await bondContract.getRemaining(tokenId)).to.equal(supply);
      expect(await bondContract.balanceOf(bondContract.getAddress(), tokenId)).to.equal(supply);
    });

    it("Should not allow non-owner to create a bond", async function () {
      await expect(
        bondContract.connect(buyer1).createBond(tokenId, price, supply, uri)
      ).to.be.revertedWithCustomError(bondContract, "OwnableUnauthorizedAccount");
    });

    it("Should not allow creation with zero price", async function () {
      await expect(
        bondContract.createBond(tokenId, 0, supply, uri)
      ).to.be.revertedWith("Price must be greater than 0");
    });

    it("Should not allow creation with zero supply", async function () {
      await expect(
        bondContract.createBond(tokenId, price, 0, uri)
      ).to.be.revertedWith("Supply must be greater than 0");
    });

    it("Should not allow creation with empty URI", async function () {
      await expect(
        bondContract.createBond(tokenId, price, supply, "")
      ).to.be.revertedWith("URI cannot be empty");
    });
  });

  describe("Token Purchasing", function () {
    const tokenId = 101;
    const price = ethers.parseEther("0.01");
    const supply = 1000;
    const uri = "ipfs://test-bond-metadata";

    beforeEach(async function () {
      await bondContract.createBond(tokenId, price, supply, uri);
    });

    it("Should allow users to buy tokens", async function () {
      const amount = 10;
      const totalCost = price * BigInt(amount);

      await expect(bondContract.connect(buyer1).buy(tokenId, amount, { value: totalCost }))
        .to.emit(bondContract, "TokensPurchased")
        .withArgs(tokenId, buyer1.address, amount, totalCost);

      expect(await bondContract.balanceOf(buyer1.address, tokenId)).to.equal(amount);
      expect(await bondContract.getRemaining(tokenId)).to.equal(supply - amount);
    });

    it("Should not allow purchase with incorrect payment", async function () {
      const amount = 10;
      const incorrectPayment = price * BigInt(amount) + ethers.parseEther("0.001");

      await expect(
        bondContract.connect(buyer1).buy(tokenId, amount, { value: incorrectPayment })
      ).to.be.revertedWith("Incorrect payment amount");
    });

    it("Should not allow purchase with insufficient supply", async function () {
      const amount = supply + 1;
      const totalCost = price * BigInt(amount);

      await expect(
        bondContract.connect(buyer1).buy(tokenId, amount, { value: totalCost })
      ).to.be.revertedWith("Insufficient supply");
    });

    it("Should not allow purchase with zero amount", async function () {
      await expect(
        bondContract.connect(buyer1).buy(tokenId, 0, { value: 0 })
      ).to.be.revertedWith("Amount must be greater than 0");
    });

    it("Should update remaining supply correctly after multiple purchases", async function () {
      const amount1 = 100;
      const amount2 = 200;
      const totalCost1 = price * BigInt(amount1);
      const totalCost2 = price * BigInt(amount2);

      await bondContract.connect(buyer1).buy(tokenId, amount1, { value: totalCost1 });
      await bondContract.connect(buyer2).buy(tokenId, amount2, { value: totalCost2 });

      expect(await bondContract.getRemaining(tokenId)).to.equal(supply - amount1 - amount2);
      expect(await bondContract.balanceOf(buyer1.address, tokenId)).to.equal(amount1);
      expect(await bondContract.balanceOf(buyer2.address, tokenId)).to.equal(amount2);
    });
  });

  describe("Fund Withdrawal", function () {
    const tokenId = 101;
    const price = ethers.parseEther("0.01");
    const supply = 1000;
    const uri = "ipfs://test-bond-metadata";

    beforeEach(async function () {
      await bondContract.createBond(tokenId, price, supply, uri);
      
      // Buy some tokens to add funds to contract
      const amount = 100;
      const totalCost = price * BigInt(amount);
      await bondContract.connect(buyer1).buy(tokenId, amount, { value: totalCost });
    });

    it("Should allow owner to withdraw funds", async function () {
      const initialBalance = await ethers.provider.getBalance(owner.address);
      const contractBalance = await ethers.provider.getBalance(bondContract.getAddress());

      await expect(bondContract.withdraw(owner.address))
        .to.emit(bondContract, "FundsWithdrawn")
        .withArgs(owner.address, contractBalance);

      expect(await ethers.provider.getBalance(bondContract.getAddress())).to.equal(0);
    });

    it("Should not allow non-owner to withdraw funds", async function () {
      await expect(
        bondContract.connect(buyer1).withdraw(buyer1.address)
      ).to.be.revertedWithCustomError(bondContract, "OwnableUnauthorizedAccount");
    });

    it("Should not allow withdrawal to zero address", async function () {
      await expect(
        bondContract.withdraw(ethers.ZeroAddress)
      ).to.be.revertedWith("Invalid recipient address");
    });

    it("Should not allow withdrawal when no funds available", async function () {
      // Withdraw all funds first
      await bondContract.withdraw(owner.address);
      
      // Try to withdraw again
      await expect(
        bondContract.withdraw(owner.address)
      ).to.be.revertedWith("No funds to withdraw");
    });
  });

  describe("Getter Functions", function () {
    const tokenId = 101;
    const price = ethers.parseEther("0.01");
    const supply = 1000;
    const uri = "ipfs://test-bond-metadata";

    beforeEach(async function () {
      await bondContract.createBond(tokenId, price, supply, uri);
    });

    it("Should return correct price", async function () {
      expect(await bondContract.getPrice(tokenId)).to.equal(price);
    });

    it("Should return correct remaining supply", async function () {
      expect(await bondContract.getRemaining(tokenId)).to.equal(supply);
    });

    it("Should return correct URI", async function () {
      expect(await bondContract.uri(tokenId)).to.equal(uri);
    });

    it("Should return zero for non-existent tokens", async function () {
      const nonExistentId = 999;
      expect(await bondContract.getPrice(nonExistentId)).to.equal(0);
      expect(await bondContract.getRemaining(nonExistentId)).to.equal(0);
    });
  });

  describe("Edge Cases", function () {
    it("Should handle multiple bond series correctly", async function () {
      const bond1 = { id: 101, price: ethers.parseEther("0.01"), supply: 1000, uri: "ipfs://bond1" };
      const bond2 = { id: 102, price: ethers.parseEther("0.02"), supply: 500, uri: "ipfs://bond2" };

      await bondContract.createBond(bond1.id, bond1.price, bond1.supply, bond1.uri);
      await bondContract.createBond(bond2.id, bond2.price, bond2.supply, bond2.uri);

      expect(await bondContract.getPrice(bond1.id)).to.equal(bond1.price);
      expect(await bondContract.getPrice(bond2.id)).to.equal(bond2.price);
      expect(await bondContract.getRemaining(bond1.id)).to.equal(bond1.supply);
      expect(await bondContract.getRemaining(bond2.id)).to.equal(bond2.supply);
    });

    it("Should prevent transfers when supply is exhausted", async function () {
      const tokenId = 101;
      const price = ethers.parseEther("0.01");
      const supply = 100;
      const uri = "ipfs://test-bond-metadata";

      await bondContract.createBond(tokenId, price, supply, uri);
      
      // Buy all tokens
      const totalCost = price * BigInt(supply);
      await bondContract.connect(buyer1).buy(tokenId, supply, { value: totalCost });

      // Try to buy more tokens
      await expect(
        bondContract.connect(buyer2).buy(tokenId, 1, { value: price })
      ).to.be.revertedWith("Insufficient supply");
    });
  });
});
