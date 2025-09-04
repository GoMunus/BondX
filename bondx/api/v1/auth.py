"""
Authentication API endpoints for BondX.

This module provides authentication endpoints including login, logout,
token refresh, and user profile management.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext

from ...core.config import settings
from ...core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = getattr(settings, 'secret_key', "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security
security = HTTPBearer()

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class UserProfile(BaseModel):
    id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    organization: str
    permissions: list
    created_at: datetime
    last_login: Optional[datetime] = None

# Mock user database (in production, this would be a real database)
MOCK_USERS = {
    "demo": {
        "id": "user_001",
        "username": "demo",
        "email": "demo@bondx.com",
        "password_hash": pwd_context.hash("demo123"),  # Password: demo123
        "first_name": "Demo",
        "last_name": "User",
        "role": "trader",
        "organization": "BondX Demo",
        "permissions": ["read", "write", "trade"],
        "created_at": datetime(2024, 1, 1),
        "is_active": True
    },
    "admin": {
        "id": "user_002",
        "username": "admin",
        "email": "admin@bondx.com",
        "password_hash": pwd_context.hash("admin123"),  # Password: admin123
        "first_name": "Admin",
        "last_name": "User",
        "role": "admin",
        "organization": "BondX",
        "permissions": ["read", "write", "trade", "admin"],
        "created_at": datetime(2024, 1, 1),
        "is_active": True
    },
    "portfolio_manager": {
        "id": "user_003",
        "username": "portfolio_manager",
        "email": "pm@bondx.com",
        "password_hash": pwd_context.hash("pm123"),  # Password: pm123
        "first_name": "Portfolio",
        "last_name": "Manager",
        "role": "portfolio_manager",
        "organization": "BondX Investment",
        "permissions": ["read", "write", "portfolio_management"],
        "created_at": datetime(2024, 1, 1),
        "is_active": True
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user with username and password."""
    user = MOCK_USERS.get(username)
    if not user or not user["is_active"]:
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    # Update last login
    user["last_login"] = datetime.utcnow()
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)  # Refresh tokens last 7 days
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return payload
    except jwt.PyJWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    user = MOCK_USERS.get(username)
    if user is None or not user["is_active"]:
        raise credentials_exception
    
    return user

@router.post("/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """
    Authenticate user and return access tokens.
    
    Demo credentials:
    - username: demo, password: demo123
    - username: admin, password: admin123
    - username: portfolio_manager, password: pm123
    """
    try:
        user = authenticate_user(login_request.username, login_request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"], "user_id": user["id"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        refresh_token = create_refresh_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )
        
        # Prepare user data (exclude sensitive information)
        user_data = {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "role": user["role"],
            "organization": user["organization"],
            "permissions": user["permissions"]
        }
        
        logger.info(f"User {user['username']} logged in successfully")
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh")
async def refresh_token(refresh_request: RefreshTokenRequest):
    """Refresh an access token using a refresh token."""
    try:
        payload = verify_token(refresh_request.refresh_token)
        if payload is None or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        username = payload.get("sub")
        user = MOCK_USERS.get(username)
        if not user or not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = create_access_token(
            data={"sub": user["username"], "user_id": user["id"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout the current user."""
    try:
        logger.info(f"User {current_user['username']} logged out")
        return {
            "success": True,
            "message": "Successfully logged out",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/profile", response_model=UserProfile)
async def get_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get the current user's profile information."""
    try:
        return UserProfile(
            id=current_user["id"],
            username=current_user["username"],
            email=current_user["email"],
            first_name=current_user["first_name"],
            last_name=current_user["last_name"],
            role=current_user["role"],
            organization=current_user["organization"],
            permissions=current_user["permissions"],
            created_at=current_user["created_at"],
            last_login=current_user.get("last_login")
        )
    except Exception as e:
        logger.error(f"Profile fetch error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user profile"
        )

@router.get("/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information (simplified endpoint)."""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"],
        "permissions": current_user["permissions"]
    }

@router.get("/verify")
async def verify_token_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Verify if the current token is valid."""
    return {
        "valid": True,
        "user_id": current_user["id"],
        "username": current_user["username"],
        "role": current_user["role"]
    }

# Export router
__all__ = ["router"]
