// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title Bond1155
 * @dev ERC-1155 contract for fractional corporate bond trading platform "BondX"
 * This contract allows the owner to create bond tokens and sell them to buyers
 * Each bond is represented by a unique token ID with its own price and supply
 */
contract Bond1155 is ERC1155, Ownable {
    
    // State variables to track bond information
    mapping(uint256 => uint256) public tokenPrice;        // Price per token in wei
    mapping(uint256 => uint256) public remainingSupply;   // Remaining tokens available for sale
    
    /**
     * @dev Constructor sets the base URI for token metadata
     * @param baseURI Base URI for all token metadata
     */
    constructor(string memory baseURI) ERC1155(baseURI) {}
    
    /**
     * @dev Creates a new bond token with specified parameters
     * @param tokenId Unique identifier for the bond series
     * @param priceWei Price per token in wei
     * @param supply Total supply of tokens for this bond
     * @param uri Metadata URI for the token
     * Only callable by the contract owner
     */
    function createBond(
        uint256 tokenId, 
        uint256 priceWei, 
        uint256 supply, 
        string memory uri
    ) external onlyOwner {
        require(priceWei > 0, "Price must be greater than 0");
        require(supply > 0, "Supply must be greater than 0");
        require(bytes(uri).length > 0, "URI cannot be empty");
        
        // Set token price and remaining supply
        tokenPrice[tokenId] = priceWei;
        remainingSupply[tokenId] = supply;
        
        // Mint tokens to the contract itself (contract acts as initial holder)
        _mint(address(this), tokenId, supply, "");
        
        // Set the token URI
        _setURI(uri);
        
        emit BondCreated(tokenId, priceWei, supply, uri);
    }
    
    /**
     * @dev Allows buyers to purchase bond tokens
     * @param tokenId ID of the bond token to purchase
     * @param amount Number of tokens to purchase
     * Requires exact payment and sufficient supply
     */
    function buy(uint256 tokenId, uint256 amount) external payable {
        require(amount > 0, "Amount must be greater than 0");
        require(remainingSupply[tokenId] >= amount, "Insufficient supply");
        
        uint256 totalCost = tokenPrice[tokenId] * amount;
        require(msg.value == totalCost, "Incorrect payment amount");
        
        // Update remaining supply
        remainingSupply[tokenId] -= amount;
        
        // Transfer tokens from contract to buyer
        _safeTransferFrom(address(this), msg.sender, tokenId, amount, "");
        
        emit TokensPurchased(tokenId, msg.sender, amount, totalCost);
    }
    
    /**
     * @dev Allows the owner to withdraw collected funds from the contract
     * @param to Address to receive the withdrawn funds
     * Only callable by the contract owner
     */
    function withdraw(address payable to) external onlyOwner {
        require(to != address(0), "Invalid recipient address");
        require(address(this).balance > 0, "No funds to withdraw");
        
        uint256 balance = address(this).balance;
        (bool success, ) = to.call{value: balance}("");
        require(success, "Withdrawal failed");
        
        emit FundsWithdrawn(to, balance);
    }
    
    /**
     * @dev Getter function for token price
     * @param tokenId ID of the bond token
     * @return Price per token in wei
     */
    function getPrice(uint256 tokenId) external view returns (uint256) {
        return tokenPrice[tokenId];
    }
    
    /**
     * @dev Getter function for remaining supply
     * @param tokenId ID of the bond token
     * @return Remaining tokens available for sale
     */
    function getRemaining(uint256 tokenId) external view returns (uint256) {
        return remainingSupply[tokenId];
    }
    
    /**
     * @dev Override the uri function to return token-specific metadata
     * @param tokenId ID of the bond token
     * @return Metadata URI for the token
     */
    function uri(uint256 tokenId) public view virtual override returns (string memory) {
        return super.uri(tokenId);
    }
    
    /**
     * @dev Override _beforeTokenTransfer to prevent transfers when supply is 0
     * @param operator Address performing the transfer
     * @param from Address tokens are transferred from
     * @param to Address tokens are transferred to
     * @param ids Array of token IDs
     * @param amounts Array of amounts
     * @param data Additional data
     */
    function _beforeTokenTransfer(
        address operator,
        address from,
        address to,
        uint256[] memory ids,
        uint256[] memory amounts,
        bytes memory data
    ) internal virtual override {
        super._beforeTokenTransfer(operator, from, to, ids, amounts, data);
        
        // Prevent transfers when remaining supply is 0 (except for initial minting)
        for (uint256 i = 0; i < ids.length; i++) {
            if (from == address(this) && remainingSupply[ids[i]] == 0) {
                revert("Token supply exhausted");
            }
        }
    }
    
    // Events for tracking important contract activities
    event BondCreated(uint256 indexed tokenId, uint256 price, uint256 supply, string uri);
    event TokensPurchased(uint256 indexed tokenId, address indexed buyer, uint256 amount, uint256 totalCost);
    event FundsWithdrawn(address indexed recipient, uint256 amount);
}
